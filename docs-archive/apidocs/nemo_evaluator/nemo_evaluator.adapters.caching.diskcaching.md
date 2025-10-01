# {py:mod}`nemo_evaluator.adapters.caching.diskcaching`

```{py:module} nemo_evaluator.adapters.caching.diskcaching
```

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching
:parser: docs.autodoc2_docstrings_parser
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Constant <nemo_evaluator.adapters.caching.diskcaching.Constant>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Constant
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`Disk <nemo_evaluator.adapters.caching.diskcaching.Disk>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Disk
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`JSONDisk <nemo_evaluator.adapters.caching.diskcaching.JSONDisk>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.JSONDisk
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`Cache <nemo_evaluator.adapters.caching.diskcaching.Cache>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Cache
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`full_name <nemo_evaluator.adapters.caching.diskcaching.full_name>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.full_name
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`args_to_key <nemo_evaluator.adapters.caching.diskcaching.args_to_key>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.args_to_key
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DBNAME <nemo_evaluator.adapters.caching.diskcaching.DBNAME>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.DBNAME
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`ENOVAL <nemo_evaluator.adapters.caching.diskcaching.ENOVAL>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.ENOVAL
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`UNKNOWN <nemo_evaluator.adapters.caching.diskcaching.UNKNOWN>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.UNKNOWN
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`MODE_NONE <nemo_evaluator.adapters.caching.diskcaching.MODE_NONE>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.MODE_NONE
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`MODE_RAW <nemo_evaluator.adapters.caching.diskcaching.MODE_RAW>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.MODE_RAW
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`MODE_JSON <nemo_evaluator.adapters.caching.diskcaching.MODE_JSON>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.MODE_JSON
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`DEFAULT_SETTINGS <nemo_evaluator.adapters.caching.diskcaching.DEFAULT_SETTINGS>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.DEFAULT_SETTINGS
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`METADATA <nemo_evaluator.adapters.caching.diskcaching.METADATA>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.METADATA
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
* - {py:obj}`EVICTION_POLICY <nemo_evaluator.adapters.caching.diskcaching.EVICTION_POLICY>`
  - ```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.EVICTION_POLICY
    :parser: docs.autodoc2_docstrings_parser
    :summary:
    ```
````

### API

````{py:function} full_name(func)
:canonical: nemo_evaluator.adapters.caching.diskcaching.full_name

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.full_name
:parser: docs.autodoc2_docstrings_parser
```
````

`````{py:class} Constant()
:canonical: nemo_evaluator.adapters.caching.diskcaching.Constant

Bases: {py:obj}`tuple`

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Constant
:parser: docs.autodoc2_docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Constant.__init__
:parser: docs.autodoc2_docstrings_parser
```

````{py:method} __new__(name)
:canonical: nemo_evaluator.adapters.caching.diskcaching.Constant.__new__

````

````{py:method} __repr__()
:canonical: nemo_evaluator.adapters.caching.diskcaching.Constant.__repr__

````

`````

````{py:data} DBNAME
:canonical: nemo_evaluator.adapters.caching.diskcaching.DBNAME
:value: >
   'cache.db'

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.DBNAME
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:data} ENOVAL
:canonical: nemo_evaluator.adapters.caching.diskcaching.ENOVAL
:value: >
   'Constant(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.ENOVAL
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:data} UNKNOWN
:canonical: nemo_evaluator.adapters.caching.diskcaching.UNKNOWN
:value: >
   'Constant(...)'

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.UNKNOWN
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:data} MODE_NONE
:canonical: nemo_evaluator.adapters.caching.diskcaching.MODE_NONE
:value: >
   0

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.MODE_NONE
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:data} MODE_RAW
:canonical: nemo_evaluator.adapters.caching.diskcaching.MODE_RAW
:value: >
   1

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.MODE_RAW
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:data} MODE_JSON
:canonical: nemo_evaluator.adapters.caching.diskcaching.MODE_JSON
:value: >
   2

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.MODE_JSON
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:data} DEFAULT_SETTINGS
:canonical: nemo_evaluator.adapters.caching.diskcaching.DEFAULT_SETTINGS
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.DEFAULT_SETTINGS
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:data} METADATA
:canonical: nemo_evaluator.adapters.caching.diskcaching.METADATA
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.METADATA
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:data} EVICTION_POLICY
:canonical: nemo_evaluator.adapters.caching.diskcaching.EVICTION_POLICY
:value: >
   None

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.EVICTION_POLICY
:parser: docs.autodoc2_docstrings_parser
```

````

`````{py:class} Disk(directory, min_file_size=0)
:canonical: nemo_evaluator.adapters.caching.diskcaching.Disk

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Disk
:parser: docs.autodoc2_docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Disk.__init__
:parser: docs.autodoc2_docstrings_parser
```

````{py:method} hash(key)
:canonical: nemo_evaluator.adapters.caching.diskcaching.Disk.hash

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Disk.hash
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} put(key)
:canonical: nemo_evaluator.adapters.caching.diskcaching.Disk.put

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Disk.put
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} get(key, raw)
:canonical: nemo_evaluator.adapters.caching.diskcaching.Disk.get

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Disk.get
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} store(value, read, key=UNKNOWN)
:canonical: nemo_evaluator.adapters.caching.diskcaching.Disk.store

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Disk.store
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _write(full_path, iterator, mode, encoding=None)
:canonical: nemo_evaluator.adapters.caching.diskcaching.Disk._write

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Disk._write
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} fetch(mode, filename, value, read)
:canonical: nemo_evaluator.adapters.caching.diskcaching.Disk.fetch

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Disk.fetch
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} filename(key=UNKNOWN, value=UNKNOWN)
:canonical: nemo_evaluator.adapters.caching.diskcaching.Disk.filename

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Disk.filename
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} remove(file_path)
:canonical: nemo_evaluator.adapters.caching.diskcaching.Disk.remove

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Disk.remove
:parser: docs.autodoc2_docstrings_parser
```

````

`````

`````{py:class} JSONDisk(directory, compress_level=1, **kwargs)
:canonical: nemo_evaluator.adapters.caching.diskcaching.JSONDisk

Bases: {py:obj}`nemo_evaluator.adapters.caching.diskcaching.Disk`

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.JSONDisk
:parser: docs.autodoc2_docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.JSONDisk.__init__
:parser: docs.autodoc2_docstrings_parser
```

````{py:method} put(key)
:canonical: nemo_evaluator.adapters.caching.diskcaching.JSONDisk.put

````

````{py:method} get(key, raw)
:canonical: nemo_evaluator.adapters.caching.diskcaching.JSONDisk.get

````

````{py:method} store(value, read, key=UNKNOWN)
:canonical: nemo_evaluator.adapters.caching.diskcaching.JSONDisk.store

````

````{py:method} fetch(mode, filename, value, read)
:canonical: nemo_evaluator.adapters.caching.diskcaching.JSONDisk.fetch

````

`````

````{py:exception} Timeout()
:canonical: nemo_evaluator.adapters.caching.diskcaching.Timeout

Bases: {py:obj}`Exception`

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Timeout
:parser: docs.autodoc2_docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Timeout.__init__
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:exception} UnknownFileWarning()
:canonical: nemo_evaluator.adapters.caching.diskcaching.UnknownFileWarning

Bases: {py:obj}`UserWarning`

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.UnknownFileWarning
:parser: docs.autodoc2_docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.UnknownFileWarning.__init__
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:exception} EmptyDirWarning()
:canonical: nemo_evaluator.adapters.caching.diskcaching.EmptyDirWarning

Bases: {py:obj}`UserWarning`

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.EmptyDirWarning
:parser: docs.autodoc2_docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.EmptyDirWarning.__init__
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:function} args_to_key(base, args, kwargs, typed, ignore)
:canonical: nemo_evaluator.adapters.caching.diskcaching.args_to_key

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.args_to_key
:parser: docs.autodoc2_docstrings_parser
```
````

`````{py:class} Cache(directory=None, timeout=60, disk=Disk, **settings)
:canonical: nemo_evaluator.adapters.caching.diskcaching.Cache

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Cache
:parser: docs.autodoc2_docstrings_parser
```

```{rubric} Initialization
```

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Cache.__init__
:parser: docs.autodoc2_docstrings_parser
```

````{py:property} directory
:canonical: nemo_evaluator.adapters.caching.diskcaching.Cache.directory

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Cache.directory
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:property} timeout
:canonical: nemo_evaluator.adapters.caching.diskcaching.Cache.timeout

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Cache.timeout
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:property} disk
:canonical: nemo_evaluator.adapters.caching.diskcaching.Cache.disk

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Cache.disk
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:property} _con
:canonical: nemo_evaluator.adapters.caching.diskcaching.Cache._con

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Cache._con
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:property} _sql
:canonical: nemo_evaluator.adapters.caching.diskcaching.Cache._sql

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Cache._sql
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:property} _sql_retry
:canonical: nemo_evaluator.adapters.caching.diskcaching.Cache._sql_retry

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Cache._sql_retry
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} transact(retry=False)
:canonical: nemo_evaluator.adapters.caching.diskcaching.Cache.transact

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Cache.transact
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _transact(retry=False, filename=None)
:canonical: nemo_evaluator.adapters.caching.diskcaching.Cache._transact

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Cache._transact
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} set(key, value, expire=None, read=False, tag=None, retry=False)
:canonical: nemo_evaluator.adapters.caching.diskcaching.Cache.set

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Cache.set
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} __setitem__(key, value)
:canonical: nemo_evaluator.adapters.caching.diskcaching.Cache.__setitem__

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Cache.__setitem__
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _row_update(rowid, now, columns)
:canonical: nemo_evaluator.adapters.caching.diskcaching.Cache._row_update

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Cache._row_update
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _row_insert(key, raw, now, columns)
:canonical: nemo_evaluator.adapters.caching.diskcaching.Cache._row_insert

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Cache._row_insert
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _cull(now, sql, cleanup, limit=None)
:canonical: nemo_evaluator.adapters.caching.diskcaching.Cache._cull

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Cache._cull
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} touch(key, expire=None, retry=False)
:canonical: nemo_evaluator.adapters.caching.diskcaching.Cache.touch

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Cache.touch
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} add(key, value, expire=None, read=False, tag=None, retry=False)
:canonical: nemo_evaluator.adapters.caching.diskcaching.Cache.add

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Cache.add
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} incr(key, delta=1, default=0, retry=False)
:canonical: nemo_evaluator.adapters.caching.diskcaching.Cache.incr

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Cache.incr
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} decr(key, delta=1, default=0, retry=False)
:canonical: nemo_evaluator.adapters.caching.diskcaching.Cache.decr

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Cache.decr
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} get(key, default=None, read=False, expire_time=False, tag=False, retry=False)
:canonical: nemo_evaluator.adapters.caching.diskcaching.Cache.get

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Cache.get
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} __getitem__(key)
:canonical: nemo_evaluator.adapters.caching.diskcaching.Cache.__getitem__

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Cache.__getitem__
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} create_tag_index()
:canonical: nemo_evaluator.adapters.caching.diskcaching.Cache.create_tag_index

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Cache.create_tag_index
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} drop_tag_index()
:canonical: nemo_evaluator.adapters.caching.diskcaching.Cache.drop_tag_index

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Cache.drop_tag_index
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} evict(tag, retry=False)
:canonical: nemo_evaluator.adapters.caching.diskcaching.Cache.evict

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Cache.evict
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} expire(now=None, retry=False)
:canonical: nemo_evaluator.adapters.caching.diskcaching.Cache.expire

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Cache.expire
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} cull(retry=False)
:canonical: nemo_evaluator.adapters.caching.diskcaching.Cache.cull

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Cache.cull
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} clear(retry=False)
:canonical: nemo_evaluator.adapters.caching.diskcaching.Cache.clear

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Cache.clear
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _select_delete(select, args, row_index=0, arg_index=0, retry=False)
:canonical: nemo_evaluator.adapters.caching.diskcaching.Cache._select_delete

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Cache._select_delete
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} iterkeys(reverse=False)
:canonical: nemo_evaluator.adapters.caching.diskcaching.Cache.iterkeys

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Cache.iterkeys
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} _iter(ascending=True)
:canonical: nemo_evaluator.adapters.caching.diskcaching.Cache._iter

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Cache._iter
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} __iter__()
:canonical: nemo_evaluator.adapters.caching.diskcaching.Cache.__iter__

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Cache.__iter__
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} __reversed__()
:canonical: nemo_evaluator.adapters.caching.diskcaching.Cache.__reversed__

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Cache.__reversed__
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} stats(enable=True, reset=False)
:canonical: nemo_evaluator.adapters.caching.diskcaching.Cache.stats

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Cache.stats
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} volume()
:canonical: nemo_evaluator.adapters.caching.diskcaching.Cache.volume

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Cache.volume
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} close()
:canonical: nemo_evaluator.adapters.caching.diskcaching.Cache.close

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Cache.close
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} __enter__()
:canonical: nemo_evaluator.adapters.caching.diskcaching.Cache.__enter__

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Cache.__enter__
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} __exit__(*exception)
:canonical: nemo_evaluator.adapters.caching.diskcaching.Cache.__exit__

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Cache.__exit__
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} __len__()
:canonical: nemo_evaluator.adapters.caching.diskcaching.Cache.__len__

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Cache.__len__
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} __getstate__()
:canonical: nemo_evaluator.adapters.caching.diskcaching.Cache.__getstate__

````

````{py:method} __setstate__(state)
:canonical: nemo_evaluator.adapters.caching.diskcaching.Cache.__setstate__

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Cache.__setstate__
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} reset(key, value=ENOVAL, update=True)
:canonical: nemo_evaluator.adapters.caching.diskcaching.Cache.reset

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Cache.reset
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} __delitem__(key)
:canonical: nemo_evaluator.adapters.caching.diskcaching.Cache.__delitem__

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Cache.__delitem__
:parser: docs.autodoc2_docstrings_parser
```

````

````{py:method} __contains__(key)
:canonical: nemo_evaluator.adapters.caching.diskcaching.Cache.__contains__

```{autodoc2-docstring} nemo_evaluator.adapters.caching.diskcaching.Cache.__contains__
:parser: docs.autodoc2_docstrings_parser
```

````

`````
