# Copyright 2016-2022 Grant Jenks
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications:
# - Removed pickle serialization in favor of raw bytes and JSON
# - Modified storage and retrieval methods to handle binary data directly

"""Core disk and file backed cache API.
source: https://github.com/grantjenks/python-diskcache
changes:
- instead of pickle, use raw bytes.

The assumption
is that we will cache requests responses, which are bytes.
"""

import codecs
import contextlib as cl
import errno
import functools as ft
import io
import json
import os
import os.path as op
import sqlite3
import tempfile
import threading
import time
import zlib


def full_name(func):
    """Return full name of `func` by adding the module and function name."""
    return func.__module__ + "." + func.__qualname__


class Constant(tuple):
    """Pretty display of immutable constant."""

    def __new__(cls, name):
        return tuple.__new__(cls, (name,))

    def __repr__(self):
        return "%s" % self[0]


DBNAME = "cache.db"
ENOVAL = Constant("ENOVAL")
UNKNOWN = Constant("UNKNOWN")

MODE_NONE = 0
MODE_RAW = 1
MODE_JSON = 2

DEFAULT_SETTINGS = {
    "statistics": 0,  # False
    "tag_index": 0,  # False
    "eviction_policy": "least-recently-stored",
    "size_limit": 2**30,  # 1gb
    "cull_limit": 10,
    "sqlite_auto_vacuum": 1,  # FULL
    "sqlite_cache_size": 2**13,  # 8,192 pages
    "sqlite_journal_mode": "wal",
    "sqlite_mmap_size": 2**26,  # 64mb
    "sqlite_synchronous": 1,  # NORMAL
    "disk_min_file_size": 2**15,  # 32kb
}

METADATA = {
    "count": 0,
    "size": 0,
    "hits": 0,
    "misses": 0,
}

EVICTION_POLICY = {
    "none": {
        "init": None,
        "get": None,
        "cull": None,
    },
    "least-recently-stored": {
        "init": ("CREATE INDEX IF NOT EXISTS Cache_store_time ON Cache (store_time)"),
        "get": None,
        "cull": "SELECT {fields} FROM Cache ORDER BY store_time LIMIT ?",
    },
    "least-recently-used": {
        "init": ("CREATE INDEX IF NOT EXISTS Cache_access_time ON Cache (access_time)"),
        "get": "access_time = {now}",
        "cull": "SELECT {fields} FROM Cache ORDER BY access_time LIMIT ?",
    },
    "least-frequently-used": {
        "init": (
            "CREATE INDEX IF NOT EXISTS Cache_access_count ON Cache (access_count)"
        ),
        "get": "access_count = access_count + 1",
        "cull": "SELECT {fields} FROM Cache ORDER BY access_count LIMIT ?",
    },
}


class Disk:
    """Cache key and value serialization for SQLite database and files."""

    def __init__(self, directory, min_file_size=0):
        """Initialize disk instance.

        :param str directory: directory path
        :param int min_file_size: minimum size for file use
        :param int json_indent: JSON indentation for serialization

        """
        self._directory = directory
        self.min_file_size = min_file_size

    def hash(self, key):
        """Compute portable hash for `key`.

        :param key: key to hash
        :return: hash value

        """
        mask = 0xFFFFFFFF
        disk_key, _ = self.put(key)
        type_disk_key = type(disk_key)

        if type_disk_key is sqlite3.Binary:
            return zlib.adler32(disk_key) & mask
        elif type_disk_key is str:
            return zlib.adler32(disk_key.encode("utf-8")) & mask
        elif type_disk_key is int:
            return disk_key % mask
        else:
            raise ValueError(f"Unsupported key type: {type_disk_key}")

    def put(self, key):
        """Convert `key` to fields key and raw for Cache table.

        :param key: key to convert
        :return: (database key, raw boolean) pair

        """
        type_key = type(key)

        if type_key is bytes:
            return sqlite3.Binary(key), True
        elif (
            (type_key is str)
            or (type_key is int and -9223372036854775808 <= key <= 9223372036854775807)
            or (type_key is float)
        ):
            return key, True
        else:
            try:
                # Try JSON serialization first
                json_data = json.dumps(key, separators=(",", ":"))
                return sqlite3.Binary(json_data.encode("utf-8")), False
            except (TypeError, ValueError):
                # Fallback to string representation if JSON fails
                return str(key), True

    def get(self, key, raw):
        """Convert fields `key` and `raw` from Cache table to key.

        :param key: database key to convert
        :param bool raw: flag indicating raw database storage
        :return: corresponding Python key

        """
        if raw:
            return bytes(key) if type(key) is sqlite3.Binary else key
        else:
            try:
                if type(key) is sqlite3.Binary:
                    return json.loads(key.decode("utf-8"))
                return json.loads(key)
            except (ValueError, TypeError):
                return key

    def store(self, value, read, key=UNKNOWN):
        """Convert `value` to fields size, mode, filename, and value for Cache
        table.

        :param value: value to convert
        :param bool read: True when value is file-like object
        :param key: key for item (default UNKNOWN)
        :return: (size, mode, filename, value) tuple for Cache table

        """
        # pylint: disable=unidiomatic-typecheck
        type_value = type(value)
        min_file_size = self.min_file_size

        if (
            (type_value is str and len(value) < min_file_size)
            or (
                type_value is int
                and -9223372036854775808 <= value <= 9223372036854775807
            )
            or (type_value is float)
        ):
            return 0, MODE_RAW, None, value
        elif type_value is bytes:
            if len(value) < min_file_size:
                return 0, MODE_RAW, None, sqlite3.Binary(value)
            else:
                filename, full_path = self.filename(key, value)
                self._write(full_path, io.BytesIO(value), "xb")
                return len(value), MODE_RAW, filename, None
        elif type_value is str:
            filename, full_path = self.filename(key, value)
            self._write(full_path, io.StringIO(value), "x", "UTF-8")
            size = op.getsize(full_path)
            return size, MODE_RAW, filename, None
        elif read:
            reader = ft.partial(value.read, 2**22)
            filename, full_path = self.filename(key, value)
            iterator = iter(reader, b"")
            size = self._write(full_path, iterator, "xb")
            return size, MODE_RAW, filename, None
        else:
            try:
                json_data = json.dumps(value, separators=(",", ":"))
                json_bytes = json_data.encode("utf-8")

                if len(json_bytes) < min_file_size:
                    return 0, MODE_RAW, None, sqlite3.Binary(json_bytes)
                else:
                    filename, full_path = self.filename(key, value)
                    self._write(full_path, io.BytesIO(json_bytes), "xb")
                    return len(json_bytes), MODE_RAW, filename, None
            except (TypeError, ValueError):
                # Fallback to string representation if JSON fails
                str_value = str(value)
                return 0, MODE_RAW, None, str_value

    def _write(self, full_path, iterator, mode, encoding=None):
        full_dir, _ = op.split(full_path)

        for count in range(1, 11):
            with cl.suppress(OSError):
                os.makedirs(full_dir)

            try:
                # Another cache may have deleted the directory before
                # the file could be opened.
                writer = open(full_path, mode, encoding=encoding)
            except OSError:
                if count == 10:
                    # Give up after 10 tries to open the file.
                    raise
                continue

            with writer:
                size = 0
                for chunk in iterator:
                    size += len(chunk)
                    writer.write(chunk)
                return size

    def fetch(self, mode, filename, value, read):
        """Convert fields `mode`, `filename`, and `value` from Cache table to
        value.

        :param int mode: value mode raw, binary, text, or pickle
        :param str filename: filename of corresponding value
        :param value: database value
        :param bool read: when True, return an open file handle
        :return: corresponding Python value
        :raises: IOError if the value cannot be read

        """
        if mode == MODE_RAW:
            if isinstance(value, sqlite3.Binary):
                try:
                    return json.loads(bytes(value).decode("utf-8"))
                except (ValueError, TypeError):
                    return bytes(value)
            return value
        elif mode == MODE_RAW:
            if read:
                return open(op.join(self._directory, filename), "rb")
            else:
                with open(op.join(self._directory, filename), "rb") as reader:
                    data = reader.read()
                    try:
                        return json.loads(data.decode("utf-8"))
                    except (ValueError, TypeError):
                        return data
        elif mode == MODE_RAW:
            full_path = op.join(self._directory, filename)
            with open(full_path, "r", encoding="UTF-8") as reader:
                return reader.read()
        else:
            # Unknown mode, return as-is
            return value

    def filename(self, key=UNKNOWN, value=UNKNOWN):
        """Return filename and full-path tuple for file storage.

        Filename will be a randomly generated 28 character hexadecimal string
        with ".val" suffixed. Two levels of sub-directories will be used to
        reduce the size of directories. On older filesystems, lookups in
        directories with many files may be slow.

        The default implementation ignores the `key` and `value` parameters.

        In some scenarios, for example :meth:`Cache.push
        <diskcache.Cache.push>`, the `key` or `value` may not be known when the
        item is stored in the cache.

        :param key: key for item (default UNKNOWN)
        :param value: value for item (default UNKNOWN)

        """
        # pylint: disable=unused-argument
        hex_name = codecs.encode(os.urandom(16), "hex").decode("utf-8")
        sub_dir = op.join(hex_name[:2], hex_name[2:4])
        name = hex_name[4:] + ".val"
        filename = op.join(sub_dir, name)
        full_path = op.join(self._directory, filename)
        return filename, full_path

    def remove(self, file_path):
        """Remove a file given by `file_path`.

        This method is cross-thread and cross-process safe. If an OSError
        occurs, it is suppressed.

        :param str file_path: relative path to file

        """
        full_path = op.join(self._directory, file_path)
        full_dir, _ = op.split(full_path)

        # Suppress OSError that may occur if two caches attempt to delete the
        # same file or directory at the same time.

        with cl.suppress(OSError):
            os.remove(full_path)

        with cl.suppress(OSError):
            os.removedirs(full_dir)


class JSONDisk(Disk):
    """Cache key and value using JSON serialization with zlib compression."""

    def __init__(self, directory, compress_level=1, **kwargs):
        """Initialize JSON disk instance.

        Keys and values are compressed using the zlib library. The
        `compress_level` is an integer from 0 to 9 controlling the level of
        compression; 1 is fastest and produces the least compression, 9 is
        slowest and produces the most compression, and 0 is no compression.

        :param str directory: directory path
        :param int compress_level: zlib compression level (default 1)
        :param kwargs: super class arguments

        """
        self.compress_level = compress_level
        super().__init__(directory, **kwargs)

    def put(self, key):
        json_bytes = json.dumps(key).encode("utf-8")
        data = zlib.compress(json_bytes, self.compress_level)
        return super().put(data)

    def get(self, key, raw):
        data = super().get(key, raw)
        print("data", data)
        return json.loads(zlib.decompress(data).decode("utf-8"))

    def store(self, value, read, key=UNKNOWN):
        if not read:
            json_bytes = json.dumps(value).encode("utf-8")
            value = zlib.compress(json_bytes, self.compress_level)
        return super().store(value, read, key=key)

    def fetch(self, mode, filename, value, read):
        data = super().fetch(mode, filename, value, read)
        if not read:
            data = json.loads(zlib.decompress(data).decode("utf-8"))
        return data


class Timeout(Exception):
    """Database timeout expired."""


class UnknownFileWarning(UserWarning):
    """Warning used by Cache.check for unknown files."""


class EmptyDirWarning(UserWarning):
    """Warning used by Cache.check for empty directories."""


def args_to_key(base, args, kwargs, typed, ignore):
    """Create cache key out of function arguments.

    :param tuple base: base of key
    :param tuple args: function arguments
    :param dict kwargs: function keyword arguments
    :param bool typed: include types in cache key
    :param set ignore: positional or keyword args to ignore
    :return: cache key tuple

    """
    args = tuple(arg for index, arg in enumerate(args) if index not in ignore)
    key = base + args + (None,)

    if kwargs:
        kwargs = {key: val for key, val in kwargs.items() if key not in ignore}
        sorted_items = sorted(kwargs.items())

        for item in sorted_items:
            key += item

    if typed:
        key += tuple(type(arg) for arg in args)

        if kwargs:
            key += tuple(type(value) for _, value in sorted_items)

    return key


class Cache:
    """Disk and file backed cache."""

    def __init__(self, directory=None, timeout=60, disk=Disk, **settings):
        """Initialize cache instance.

        :param str directory: cache directory
        :param float timeout: SQLite connection timeout
        :param disk: Disk type or subclass for serialization
        :param settings: any of DEFAULT_SETTINGS

        """
        try:
            assert issubclass(disk, Disk)
        except (TypeError, AssertionError):
            raise ValueError("disk must subclass diskcache.Disk") from None

        if directory is None:
            directory = tempfile.mkdtemp(prefix="diskcache-")
        directory = str(directory)
        directory = op.expanduser(directory)
        directory = op.expandvars(directory)

        self._directory = directory
        self._timeout = 0  # Manually handle retries during initialization.
        self._local = threading.local()
        self._txn_id = None

        if not op.isdir(directory):
            try:
                os.makedirs(directory, 0o755)
            except OSError as error:
                if error.errno != errno.EEXIST:
                    raise EnvironmentError(
                        error.errno,
                        'Cache directory "%s" does not exist'
                        " and could not be created" % self._directory,
                    ) from None

        sql = self._sql_retry

        # Setup Settings table.

        try:
            current_settings = dict(sql("SELECT key, value FROM Settings").fetchall())
        except sqlite3.OperationalError:
            current_settings = {}

        sets = DEFAULT_SETTINGS.copy()
        sets.update(current_settings)
        sets.update(settings)

        for key in METADATA:
            sets.pop(key, None)

        # Chance to set pragmas before any tables are created.

        for key, value in sorted(sets.items()):
            if key.startswith("sqlite_"):
                self.reset(key, value, update=False)

        sql("CREATE TABLE IF NOT EXISTS Settings ( key TEXT NOT NULL UNIQUE, value)")

        # Setup Disk object (must happen after settings initialized).

        kwargs = {
            key[5:]: value for key, value in sets.items() if key.startswith("disk_")
        }
        self._disk = disk(directory, **kwargs)

        # Set cached attributes: updates settings and sets pragmas.

        for key, value in sets.items():
            query = "INSERT OR REPLACE INTO Settings VALUES (?, ?)"
            sql(query, (key, value))
            self.reset(key, value)

        for key, value in METADATA.items():
            query = "INSERT OR IGNORE INTO Settings VALUES (?, ?)"
            sql(query, (key, value))
            self.reset(key)

        ((self._page_size,),) = sql("PRAGMA page_size").fetchall()

        # Setup Cache table.

        sql(
            "CREATE TABLE IF NOT EXISTS Cache ("
            " rowid INTEGER PRIMARY KEY,"
            " key BLOB,"
            " raw INTEGER,"
            " store_time REAL,"
            " expire_time REAL,"
            " access_time REAL,"
            " access_count INTEGER DEFAULT 0,"
            " tag BLOB,"
            " size INTEGER DEFAULT 0,"
            " mode INTEGER DEFAULT 0,"
            " filename TEXT,"
            " value BLOB)"
        )

        sql("CREATE UNIQUE INDEX IF NOT EXISTS Cache_key_raw ON Cache(key, raw)")

        sql("CREATE INDEX IF NOT EXISTS Cache_expire_time ON Cache (expire_time)")

        query = EVICTION_POLICY[self.eviction_policy]["init"]

        if query is not None:
            sql(query)

        # Use triggers to keep Metadata updated.

        sql(
            "CREATE TRIGGER IF NOT EXISTS Settings_count_insert"
            " AFTER INSERT ON Cache FOR EACH ROW BEGIN"
            " UPDATE Settings SET value = value + 1"
            ' WHERE key = "count"; END'
        )

        sql(
            "CREATE TRIGGER IF NOT EXISTS Settings_count_delete"
            " AFTER DELETE ON Cache FOR EACH ROW BEGIN"
            " UPDATE Settings SET value = value - 1"
            ' WHERE key = "count"; END'
        )

        sql(
            "CREATE TRIGGER IF NOT EXISTS Settings_size_insert"
            " AFTER INSERT ON Cache FOR EACH ROW BEGIN"
            " UPDATE Settings SET value = value + NEW.size"
            ' WHERE key = "size"; END'
        )

        sql(
            "CREATE TRIGGER IF NOT EXISTS Settings_size_update"
            " AFTER UPDATE ON Cache FOR EACH ROW BEGIN"
            " UPDATE Settings"
            " SET value = value + NEW.size - OLD.size"
            ' WHERE key = "size"; END'
        )

        sql(
            "CREATE TRIGGER IF NOT EXISTS Settings_size_delete"
            " AFTER DELETE ON Cache FOR EACH ROW BEGIN"
            " UPDATE Settings SET value = value - OLD.size"
            ' WHERE key = "size"; END'
        )

        # Create tag index if requested.

        if self.tag_index:  # pylint: disable=no-member
            self.create_tag_index()
        else:
            self.drop_tag_index()

        # Close and re-open database connection with given timeout.

        self.close()
        self._timeout = timeout
        self._sql  # pylint: disable=pointless-statement

    @property
    def directory(self):
        """Cache directory."""
        return self._directory

    @property
    def timeout(self):
        """SQLite connection timeout value in seconds."""
        return self._timeout

    @property
    def disk(self):
        """Disk used for serialization."""
        return self._disk

    @property
    def _con(self):
        # Check process ID to support process forking. If the process
        # ID changes, close the connection and update the process ID.

        local_pid = getattr(self._local, "pid", None)
        pid = os.getpid()

        if local_pid != pid:
            self.close()
            self._local.pid = pid

        con = getattr(self._local, "con", None)

        if con is None:
            con = self._local.con = sqlite3.connect(
                op.join(self._directory, DBNAME),
                timeout=self._timeout,
                isolation_level=None,
            )

            # Some SQLite pragmas work on a per-connection basis so
            # query the Settings table and reset the pragmas. The
            # Settings table may not exist so catch and ignore the
            # OperationalError that may occur.

            try:
                select = "SELECT key, value FROM Settings"
                settings = con.execute(select).fetchall()
            except sqlite3.OperationalError:
                pass
            else:
                for key, value in settings:
                    if key.startswith("sqlite_"):
                        self.reset(key, value, update=False)

        return con

    @property
    def _sql(self):
        return self._con.execute

    @property
    def _sql_retry(self):
        sql = self._sql

        # 2018-11-01 GrantJ - Some SQLite builds/versions handle
        # the SQLITE_BUSY return value and connection parameter
        # "timeout" differently. For a more reliable duration,
        # manually retry the statement for 60 seconds. Only used
        # by statements which modify the database and do not use
        # a transaction (like those in ``__init__`` or ``reset``).
        # See Issue #85 for and tests/issue_85.py for more details.

        def _execute_with_retry(statement, *args, **kwargs):
            start = time.time()
            while True:
                try:
                    return sql(statement, *args, **kwargs)
                except sqlite3.OperationalError as exc:
                    if str(exc) != "database is locked":
                        raise
                    diff = time.time() - start
                    if diff > 60:
                        raise
                    time.sleep(0.001)

        return _execute_with_retry

    @cl.contextmanager
    def transact(self, retry=False):
        """Context manager to perform a transaction by locking the cache.

        While the cache is locked, no other write operation is permitted.
        Transactions should therefore be as short as possible. Read and write
        operations performed in a transaction are atomic. Read operations may
        occur concurrent to a transaction.

        Transactions may be nested and may not be shared between threads.

        Raises :exc:`Timeout` error when database timeout occurs and `retry` is
        `False` (default).

        >>> cache = Cache()
        >>> with cache.transact():  # Atomically increment two keys.
        ...     _ = cache.incr('total', 123.4)
        ...     _ = cache.incr('count', 1)
        >>> with cache.transact():  # Atomically calculate average.
        ...     average = cache['total'] / cache['count']
        >>> average
        123.4

        :param bool retry: retry if database timeout occurs (default False)
        :return: context manager for use in `with` statement
        :raises Timeout: if database timeout occurs

        """
        with self._transact(retry=retry):
            yield

    @cl.contextmanager
    def _transact(self, retry=False, filename=None):
        sql = self._sql
        filenames = []
        _disk_remove = self._disk.remove
        tid = threading.get_ident()
        txn_id = self._txn_id

        if tid == txn_id:
            begin = False
        else:
            while True:
                try:
                    sql("BEGIN IMMEDIATE")
                    begin = True
                    self._txn_id = tid
                    break
                except sqlite3.OperationalError:
                    if retry:
                        continue
                    if filename is not None:
                        _disk_remove(filename)
                    raise Timeout from None

        try:
            yield sql, filenames.append
        except BaseException:
            if begin:
                assert self._txn_id == tid
                self._txn_id = None
                sql("ROLLBACK")
            raise
        else:
            if begin:
                assert self._txn_id == tid
                self._txn_id = None
                sql("COMMIT")
            for name in filenames:
                if name is not None:
                    _disk_remove(name)

    def set(self, key, value, expire=None, read=False, tag=None, retry=False):
        """Set `key` and `value` item in cache.

        When `read` is `True`, `value` should be a file-like object opened
        for reading in binary mode.

        Raises :exc:`Timeout` error when database timeout occurs and `retry` is
        `False` (default).

        :param key: key for item
        :param value: value for item
        :param float expire: seconds until item expires
            (default None, no expiry)
        :param bool read: read value as bytes from file (default False)
        :param str tag: text to associate with key (default None)
        :param bool retry: retry if database timeout occurs (default False)
        :return: True if item was set
        :raises Timeout: if database timeout occurs

        """
        now = time.time()
        db_key, raw = self._disk.put(key)
        expire_time = None if expire is None else now + expire
        size, mode, filename, db_value = self._disk.store(value, read, key=key)
        columns = (expire_time, tag, size, mode, filename, db_value)

        # The order of SELECT, UPDATE, and INSERT is important below.
        #
        # Typical cache usage pattern is:
        #
        # value = cache.get(key)
        # if value is None:
        #     value = expensive_calculation()
        #     cache.set(key, value)
        #
        # Cache.get does not evict expired keys to avoid writes during lookups.
        # Commonly used/expired keys will therefore remain in the cache making
        # an UPDATE the preferred path.
        #
        # The alternative is to assume the key is not present by first trying
        # to INSERT and then handling the IntegrityError that occurs from
        # violating the UNIQUE constraint. This optimistic approach was
        # rejected based on the common cache usage pattern.
        #
        # INSERT OR REPLACE aka UPSERT is not used because the old filename may
        # need cleanup.

        with self._transact(retry, filename) as (sql, cleanup):
            rows = sql(
                "SELECT rowid, filename FROM Cache WHERE key = ? AND raw = ?",
                (db_key, raw),
            ).fetchall()

            if rows:
                ((rowid, old_filename),) = rows
                cleanup(old_filename)
                self._row_update(rowid, now, columns)
            else:
                self._row_insert(db_key, raw, now, columns)

            self._cull(now, sql, cleanup)

            return True

    def __setitem__(self, key, value):
        """Set `value` for `key` in cache."""
        # Convert key to string if needed
        if not isinstance(key, str):
            key = str(key)

        # Store value based on its type
        if isinstance(value, bytes):
            mode = MODE_RAW
            db_value = sqlite3.Binary(value)  # Store bytes using Binary wrapper
        else:
            mode = MODE_JSON
            # Convert non-bytes values to JSON string and then to bytes
            db_value = sqlite3.Binary(json.dumps(value).encode())

        # Store in database
        statement = (
            "INSERT OR REPLACE INTO Cache "
            "(key, raw, value, filename, mode, size, expire_time) "
            "VALUES (?, ?, ?, NULL, ?, ?, NULL)"
        )

        self._sql(statement, (key, True, db_value, mode, len(bytes(db_value))))

    def _row_update(self, rowid, now, columns):
        sql = self._sql
        expire_time, tag, size, mode, filename, value = columns
        sql(
            "UPDATE Cache SET"
            " store_time = ?,"
            " expire_time = ?,"
            " access_time = ?,"
            " access_count = ?,"
            " tag = ?,"
            " size = ?,"
            " mode = ?,"
            " filename = ?,"
            " value = ?"
            " WHERE rowid = ?",
            (
                now,  # store_time
                expire_time,
                now,  # access_time
                0,  # access_count
                tag,
                size,
                mode,
                filename,
                value,
                rowid,
            ),
        )

    def _row_insert(self, key, raw, now, columns):
        sql = self._sql
        expire_time, tag, size, mode, filename, value = columns
        sql(
            "INSERT INTO Cache("
            " key, raw, store_time, expire_time, access_time,"
            " access_count, tag, size, mode, filename, value"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                key,
                raw,
                now,  # store_time
                expire_time,
                now,  # access_time
                0,  # access_count
                tag,
                size,
                mode,
                filename,
                value,
            ),
        )

    def _cull(self, now, sql, cleanup, limit=None):
        cull_limit = self.cull_limit if limit is None else limit

        if cull_limit == 0:
            return

        # Evict expired keys.

        select_expired_template = (
            "SELECT %s FROM Cache"
            " WHERE expire_time IS NOT NULL AND expire_time < ?"
            " ORDER BY expire_time LIMIT ?"
        )

        select_expired = select_expired_template % "filename"
        rows = sql(select_expired, (now, cull_limit)).fetchall()

        if rows:
            delete_expired = "DELETE FROM Cache WHERE rowid IN (%s)" % (
                select_expired_template % "rowid"
            )
            sql(delete_expired, (now, cull_limit))

            for (filename,) in rows:
                cleanup(filename)

            cull_limit -= len(rows)

            if cull_limit == 0:
                return

        # Evict keys by policy.

        select_policy = EVICTION_POLICY[self.eviction_policy]["cull"]

        if select_policy is None or self.volume() < self.size_limit:
            return

        select_filename = select_policy.format(fields="filename", now=now)
        rows = sql(select_filename, (cull_limit,)).fetchall()

        if rows:
            delete = "DELETE FROM Cache WHERE rowid IN (%s)" % (
                select_policy.format(fields="rowid", now=now)
            )
            sql(delete, (cull_limit,))

            for (filename,) in rows:
                cleanup(filename)

    def touch(self, key, expire=None, retry=False):
        """Touch `key` in cache and update `expire` time.

        Raises :exc:`Timeout` error when database timeout occurs and `retry` is
        `False` (default).

        :param key: key for item
        :param float expire: seconds until item expires
            (default None, no expiry)
        :param bool retry: retry if database timeout occurs (default False)
        :return: True if key was touched
        :raises Timeout: if database timeout occurs

        """
        now = time.time()
        db_key, raw = self._disk.put(key)
        expire_time = None if expire is None else now + expire

        with self._transact(retry) as (sql, _):
            rows = sql(
                "SELECT rowid, expire_time FROM Cache WHERE key = ? AND raw = ?",
                (db_key, raw),
            ).fetchall()

            if rows:
                ((rowid, old_expire_time),) = rows

                if old_expire_time is None or old_expire_time > now:
                    sql(
                        "UPDATE Cache SET expire_time = ? WHERE rowid = ?",
                        (expire_time, rowid),
                    )
                    return True

        return False

    def add(self, key, value, expire=None, read=False, tag=None, retry=False):
        """Add `key` and `value` item to cache.

        Similar to `set`, but only add to cache if key not present.

        Operation is atomic. Only one concurrent add operation for a given key
        will succeed.

        When `read` is `True`, `value` should be a file-like object opened
        for reading in binary mode.

        Raises :exc:`Timeout` error when database timeout occurs and `retry` is
        `False` (default).

        :param key: key for item
        :param value: value for item
        :param float expire: seconds until the key expires
            (default None, no expiry)
        :param bool read: read value as bytes from file (default False)
        :param str tag: text to associate with key (default None)
        :param bool retry: retry if database timeout occurs (default False)
        :return: True if item was added
        :raises Timeout: if database timeout occurs

        """
        now = time.time()
        db_key, raw = self._disk.put(key)
        expire_time = None if expire is None else now + expire
        size, mode, filename, db_value = self._disk.store(value, read, key=key)
        columns = (expire_time, tag, size, mode, filename, db_value)

        with self._transact(retry, filename) as (sql, cleanup):
            rows = sql(
                "SELECT rowid, filename, expire_time FROM Cache"
                " WHERE key = ? AND raw = ?",
                (db_key, raw),
            ).fetchall()

            if rows:
                ((rowid, old_filename, old_expire_time),) = rows

                if old_expire_time is None or old_expire_time > now:
                    cleanup(filename)
                    return False

                cleanup(old_filename)
                self._row_update(rowid, now, columns)
            else:
                self._row_insert(db_key, raw, now, columns)

            self._cull(now, sql, cleanup)

            return True

    def incr(self, key, delta=1, default=0, retry=False):
        """Increment value by delta for item with key.

        If key is missing and default is None then raise KeyError. Else if key
        is missing and default is not None then use default for value.

        Operation is atomic. All concurrent increment operations will be
        counted individually.

        Assumes value may be stored in a SQLite column. Most builds that target
        machines with 64-bit pointer widths will support 64-bit signed
        integers.

        Raises :exc:`Timeout` error when database timeout occurs and `retry` is
        `False` (default).

        :param key: key for item
        :param int delta: amount to increment (default 1)
        :param int default: value if key is missing (default 0)
        :param bool retry: retry if database timeout occurs (default False)
        :return: new value for item
        :raises KeyError: if key is not found and default is None
        :raises Timeout: if database timeout occurs

        """
        now = time.time()
        db_key, raw = self._disk.put(key)
        select = (
            "SELECT rowid, expire_time, filename, value FROM Cache"
            " WHERE key = ? AND raw = ?"
        )

        with self._transact(retry) as (sql, cleanup):
            rows = sql(select, (db_key, raw)).fetchall()

            if not rows:
                if default is None:
                    raise KeyError(key)

                value = default + delta
                columns = (None, None) + self._disk.store(value, False, key=key)
                self._row_insert(db_key, raw, now, columns)
                self._cull(now, sql, cleanup)
                return value

            ((rowid, expire_time, filename, value),) = rows

            if expire_time is not None and expire_time < now:
                if default is None:
                    raise KeyError(key)

                value = default + delta
                columns = (None, None) + self._disk.store(value, False, key=key)
                self._row_update(rowid, now, columns)
                self._cull(now, sql, cleanup)
                cleanup(filename)
                return value

            value += delta

            columns = "store_time = ?, value = ?"
            update_column = EVICTION_POLICY[self.eviction_policy]["get"]

            if update_column is not None:
                columns += ", " + update_column.format(now=now)

            update = "UPDATE Cache SET %s WHERE rowid = ?" % columns
            sql(update, (now, value, rowid))

            return value

    def decr(self, key, delta=1, default=0, retry=False):
        """Decrement value by delta for item with key.

        If key is missing and default is None then raise KeyError. Else if key
        is missing and default is not None then use default for value.

        Operation is atomic. All concurrent decrement operations will be
        counted individually.

        Unlike Memcached, negative values are supported. Value may be
        decremented below zero.

        Assumes value may be stored in a SQLite column. Most builds that target
        machines with 64-bit pointer widths will support 64-bit signed
        integers.

        Raises :exc:`Timeout` error when database timeout occurs and `retry` is
        `False` (default).

        :param key: key for item
        :param int delta: amount to decrement (default 1)
        :param int default: value if key is missing (default 0)
        :param bool retry: retry if database timeout occurs (default False)
        :return: new value for item
        :raises KeyError: if key is not found and default is None
        :raises Timeout: if database timeout occurs

        """
        return self.incr(key, -delta, default, retry)

    def get(
        self,
        key,
        default=None,
        read=False,
        expire_time=False,
        tag=False,
        retry=False,
    ):
        """Retrieve value from cache. If `key` is missing, return `default`."""
        db_key, raw = self._disk.put(key)
        update_column = EVICTION_POLICY[self.eviction_policy]["get"]
        select = (
            "SELECT rowid, expire_time, tag, mode, filename, value"
            " FROM Cache WHERE key = ? AND raw = ?"
            " AND (expire_time IS NULL OR expire_time > ?)"
        )

        if expire_time and tag:
            default = (default, None, None)
        elif expire_time or tag:
            default = (default, None)

        if not self.statistics and update_column is None:
            # Fast path, no transaction necessary.
            rows = self._sql(select, (db_key, raw, time.time())).fetchall()

            if not rows:
                return default

            ((rowid, db_expire_time, db_tag, mode, filename, db_value),) = rows

            try:
                value = self._disk.fetch(mode, filename, db_value, read)

                # Handle JSON parsing
                if isinstance(value, (str, bytes)):
                    try:
                        if isinstance(value, bytes):
                            value = value.decode("utf-8")
                        return json.loads(value)
                    except json.JSONDecodeError:
                        return value
                return value
            except IOError:
                return default

        else:  # Slow path, transaction required.
            cache_miss = 'UPDATE Settings SET value = value + 1 WHERE key = "misses"'

            with self._transact(retry) as (sql, _):
                rows = sql(select, (db_key, raw, time.time())).fetchall()

                if not rows:
                    if self.statistics:
                        sql(cache_miss)
                    return default

                ((rowid, db_expire_time, db_tag, mode, filename, db_value),) = rows

                try:
                    value = self._disk.fetch(mode, filename, db_value, read)
                    print(value)
                    # Handle JSON parsing
                    if isinstance(value, (str, bytes)):
                        if isinstance(value, bytes):
                            value = value.decode("utf-8")
                        return json.loads(value)
                    return value
                except IOError:
                    if self.statistics:
                        sql(cache_miss)
                    return default

    def __getitem__(self, key):
        """Return corresponding value for `key` from cache."""
        # Convert key to string if needed
        if not isinstance(key, str):
            key = str(key)

        select = (
            "SELECT mode, value FROM Cache "
            "WHERE key = ? AND raw = ? "
            "AND (expire_time IS NULL OR expire_time > ?)"
        )

        rows = self._sql(select, (key, True, time.time())).fetchall()
        if not rows:
            raise KeyError(key)

        mode, value = rows[0]

        # Return value based on mode
        if mode == MODE_RAW:
            return bytes(value)  # Convert Binary back to bytes
        elif mode == MODE_JSON:
            return json.loads(bytes(value).decode())  # Parse JSON
        else:
            raise ValueError(f"Unknown storage mode: {mode}")

    def create_tag_index(self):
        """Create tag index on cache database.

        It is better to initialize cache with `tag_index=True` than use this.

        :raises Timeout: if database timeout occurs

        """
        sql = self._sql
        sql("CREATE INDEX IF NOT EXISTS Cache_tag_rowid ON Cache(tag, rowid)")
        self.reset("tag_index", 1)

    def drop_tag_index(self):
        """Drop tag index on cache database.

        :raises Timeout: if database timeout occurs

        """
        sql = self._sql
        sql("DROP INDEX IF EXISTS Cache_tag_rowid")
        self.reset("tag_index", 0)

    def evict(self, tag, retry=False):
        """Remove items with matching `tag` from cache.

        Removing items is an iterative process. In each iteration, a subset of
        items is removed. Concurrent writes may occur between iterations.

        If a :exc:`Timeout` occurs, the first element of the exception's
        `args` attribute will be the number of items removed before the
        exception occurred.

        Raises :exc:`Timeout` error when database timeout occurs and `retry` is
        `False` (default).

        :param str tag: tag identifying items
        :param bool retry: retry if database timeout occurs (default False)
        :return: count of rows removed
        :raises Timeout: if database timeout occurs

        """
        select = (
            "SELECT rowid, filename FROM Cache"
            " WHERE tag = ? AND rowid > ?"
            " ORDER BY rowid LIMIT ?"
        )
        args = [tag, 0, 100]
        return self._select_delete(select, args, arg_index=1, retry=retry)

    def expire(self, now=None, retry=False):
        """Remove expired items from cache.

        Removing items is an iterative process. In each iteration, a subset of
        items is removed. Concurrent writes may occur between iterations.

        If a :exc:`Timeout` occurs, the first element of the exception's
        `args` attribute will be the number of items removed before the
        exception occurred.

        Raises :exc:`Timeout` error when database timeout occurs and `retry` is
        `False` (default).

        :param float now: current time (default None, ``time.time()`` used)
        :param bool retry: retry if database timeout occurs (default False)
        :return: count of items removed
        :raises Timeout: if database timeout occurs

        """
        select = (
            "SELECT rowid, expire_time, filename FROM Cache"
            " WHERE ? < expire_time AND expire_time < ?"
            " ORDER BY expire_time LIMIT ?"
        )
        args = [0, now or time.time(), 100]
        return self._select_delete(select, args, row_index=1, retry=retry)

    def cull(self, retry=False):
        """Cull items from cache until volume is less than size limit.

        Removing items is an iterative process. In each iteration, a subset of
        items is removed. Concurrent writes may occur between iterations.

        If a :exc:`Timeout` occurs, the first element of the exception's
        `args` attribute will be the number of items removed before the
        exception occurred.

        Raises :exc:`Timeout` error when database timeout occurs and `retry` is
        `False` (default).

        :param bool retry: retry if database timeout occurs (default False)
        :return: count of items removed
        :raises Timeout: if database timeout occurs

        """
        now = time.time()

        # Remove expired items.

        count = self.expire(now)

        # Remove items by policy.

        select_policy = EVICTION_POLICY[self.eviction_policy]["cull"]

        if select_policy is None:
            return 0

        select_filename = select_policy.format(fields="filename", now=now)

        try:
            while self.volume() > self.size_limit:
                with self._transact(retry) as (sql, cleanup):
                    rows = sql(select_filename, (10,)).fetchall()

                    if not rows:
                        break

                    count += len(rows)
                    delete = (
                        "DELETE FROM Cache WHERE rowid IN (%s)"
                        % select_policy.format(fields="rowid", now=now)
                    )
                    sql(delete, (10,))

                    for (filename,) in rows:
                        cleanup(filename)
        except Timeout:
            raise Timeout(count) from None

        return count

    def clear(self, retry=False):
        """Remove all items from cache.

        Removing items is an iterative process. In each iteration, a subset of
        items is removed. Concurrent writes may occur between iterations.

        If a :exc:`Timeout` occurs, the first element of the exception's
        `args` attribute will be the number of items removed before the
        exception occurred.

        Raises :exc:`Timeout` error when database timeout occurs and `retry` is
        `False` (default).

        :param bool retry: retry if database timeout occurs (default False)
        :return: count of rows removed
        :raises Timeout: if database timeout occurs

        """
        select = (
            "SELECT rowid, filename FROM Cache WHERE rowid > ? ORDER BY rowid LIMIT ?"
        )
        args = [0, 100]
        return self._select_delete(select, args, retry=retry)

    def _select_delete(self, select, args, row_index=0, arg_index=0, retry=False):
        count = 0
        delete = "DELETE FROM Cache WHERE rowid IN (%s)"

        try:
            while True:
                with self._transact(retry) as (sql, cleanup):
                    rows = sql(select, args).fetchall()

                    if not rows:
                        break

                    count += len(rows)
                    sql(delete % ",".join(str(row[0]) for row in rows))

                    for row in rows:
                        args[arg_index] = row[row_index]
                        cleanup(row[-1])

        except Timeout:
            raise Timeout(count) from None

        return count

    def iterkeys(self, reverse=False):
        """Iterate Cache keys in database sort order.

        >>> cache = Cache()
        >>> for key in [4, 1, 3, 0, 2]:
        ...     cache[key] = key
        >>> list(cache.iterkeys())
        [0, 1, 2, 3, 4]
        >>> list(cache.iterkeys(reverse=True))
        [4, 3, 2, 1, 0]

        :param bool reverse: reverse sort order (default False)
        :return: iterator of Cache keys

        """
        sql = self._sql
        limit = 100
        _disk_get = self._disk.get

        if reverse:
            select = "SELECT key, raw FROM Cache ORDER BY key DESC, raw DESC LIMIT 1"
            iterate = (
                "SELECT key, raw FROM Cache"
                " WHERE key = ? AND raw < ? OR key < ?"
                " ORDER BY key DESC, raw DESC LIMIT ?"
            )
        else:
            select = "SELECT key, raw FROM Cache ORDER BY key ASC, raw ASC LIMIT 1"
            iterate = (
                "SELECT key, raw FROM Cache"
                " WHERE key = ? AND raw > ? OR key > ?"
                " ORDER BY key ASC, raw ASC LIMIT ?"
            )

        row = sql(select).fetchall()

        if row:
            ((key, raw),) = row
        else:
            return

        yield _disk_get(key, raw)

        while True:
            rows = sql(iterate, (key, raw, key, limit)).fetchall()

            if not rows:
                break

            for key, raw in rows:
                yield _disk_get(key, raw)

    def _iter(self, ascending=True):
        sql = self._sql
        rows = sql("SELECT MAX(rowid) FROM Cache").fetchall()
        ((max_rowid,),) = rows
        yield  # Signal ready.

        if max_rowid is None:
            return

        bound = max_rowid + 1
        limit = 100
        _disk_get = self._disk.get
        rowid = 0 if ascending else bound
        select = (
            "SELECT rowid, key, raw FROM Cache"
            " WHERE ? < rowid AND rowid < ?"
            " ORDER BY rowid %s LIMIT ?"
        ) % ("ASC" if ascending else "DESC")

        while True:
            if ascending:
                args = (rowid, bound, limit)
            else:
                args = (0, rowid, limit)

            rows = sql(select, args).fetchall()

            if not rows:
                break

            for rowid, key, raw in rows:
                yield _disk_get(key, raw)

    def __iter__(self):
        """Iterate keys in cache including expired items."""
        iterator = self._iter()
        next(iterator)
        return iterator

    def __reversed__(self):
        """Reverse iterate keys in cache including expired items."""
        iterator = self._iter(ascending=False)
        next(iterator)
        return iterator

    def stats(self, enable=True, reset=False):
        """Return cache statistics hits and misses.

        :param bool enable: enable collecting statistics (default True)
        :param bool reset: reset hits and misses to 0 (default False)
        :return: (hits, misses)

        """
        # pylint: disable=E0203,W0201
        result = (self.reset("hits"), self.reset("misses"))

        if reset:
            self.reset("hits", 0)
            self.reset("misses", 0)

        self.reset("statistics", enable)

        return result

    def volume(self):
        """Return estimated total size of cache on disk.

        :return: size in bytes

        """
        ((page_count,),) = self._sql("PRAGMA page_count").fetchall()
        total_size = self._page_size * page_count + self.reset("size")
        return total_size

    def close(self):
        """Close database connection."""
        con = getattr(self._local, "con", None)

        if con is None:
            return

        con.close()

        try:
            delattr(self._local, "con")
        except AttributeError:
            pass

    def __enter__(self):
        # Create connection in thread.
        # pylint: disable=unused-variable
        connection = self._con  # noqa
        return self

    def __exit__(self, *exception):
        self.close()

    def __len__(self):
        """Return number of items in cache."""
        select = (
            "SELECT COUNT(*) FROM Cache "
            "WHERE raw = ? AND (expire_time IS NULL OR expire_time > ?)"
        )

        return self._sql(select, (True, time.time())).fetchone()[0]

    def __getstate__(self):
        return (self.directory, self.timeout, type(self.disk))

    def __setstate__(self, state):
        self.__init__(*state)

    def reset(self, key, value=ENOVAL, update=True):
        """Reset `key` and `value` item from Settings table.

        Use `reset` to update the value of Cache settings correctly. Cache
        settings are stored in the Settings table of the SQLite database. If
        `update` is ``False`` then no attempt is made to update the database.

        If `value` is not given, it is reloaded from the Settings
        table. Otherwise, the Settings table is updated.

        Settings with the ``disk_`` prefix correspond to Disk
        attributes. Updating the value will change the unprefixed attribute on
        the associated Disk instance.

        Settings with the ``sqlite_`` prefix correspond to SQLite
        pragmas. Updating the value will execute the corresponding PRAGMA
        statement.

        SQLite PRAGMA statements may be executed before the Settings table
        exists in the database by setting `update` to ``False``.

        :param str key: Settings key for item
        :param value: value for item (optional)
        :param bool update: update database Settings table (default True)
        :return: updated value for item
        :raises Timeout: if database timeout occurs

        """
        sql = self._sql
        sql_retry = self._sql_retry

        if value is ENOVAL:
            select = "SELECT value FROM Settings WHERE key = ?"
            ((value,),) = sql_retry(select, (key,)).fetchall()
            setattr(self, key, value)
            return value

        if update:
            statement = "UPDATE Settings SET value = ? WHERE key = ?"
            sql_retry(statement, (value, key))

        if key.startswith("sqlite_"):
            pragma = key[7:]

            # 2016-02-17 GrantJ - PRAGMA and isolation_level=None
            # don't always play nicely together. Retry setting the
            # PRAGMA. I think some PRAGMA statements expect to
            # immediately take an EXCLUSIVE lock on the database. I
            # can't find any documentation for this but without the
            # retry, stress will intermittently fail with multiple
            # processes.

            # 2018-11-05 GrantJ - Avoid setting pragma values that
            # are already set. Pragma settings like auto_vacuum and
            # journal_mode can take a long time or may not work after
            # tables have been created.

            start = time.time()
            while True:
                try:
                    try:
                        ((old_value,),) = sql("PRAGMA %s" % (pragma)).fetchall()
                        update = old_value != value
                    except ValueError:
                        update = True
                    if update:
                        sql("PRAGMA %s = %s" % (pragma, value)).fetchall()
                    break
                except sqlite3.OperationalError as exc:
                    if str(exc) != "database is locked":
                        raise
                    diff = time.time() - start
                    if diff > 60:
                        raise
                    time.sleep(0.001)
        elif key.startswith("disk_"):
            attr = key[5:]
            setattr(self._disk, attr, value)

        setattr(self, key, value)
        return value

    def __delitem__(self, key):
        """Delete corresponding item for `key` from cache."""
        # Convert key to string if needed
        if not isinstance(key, str):
            key = str(key)

        statement = "DELETE FROM Cache WHERE key = ? AND raw = ?"
        result = self._sql(statement, (key, True))

        if result.rowcount == 0:
            raise KeyError(key)

    def __contains__(self, key):
        """Return True if `key` in cache."""
        # Convert key to string if needed
        if not isinstance(key, str):
            key = str(key)

        select = (
            "SELECT COUNT(*) FROM Cache "
            "WHERE key = ? AND raw = ? "
            "AND (expire_time IS NULL OR expire_time > ?)"
        )

        return self._sql(select, (key, True, time.time())).fetchone()[0] > 0
