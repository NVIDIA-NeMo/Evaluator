"""
Automatically generate API documentation with intelligent Pydantic model detection.

This extension scans specified packages and generates .rst files that use:
- autopydantic_model for Pydantic BaseModel classes
- autoclass for regular Python classes
- automodule for modules

Creates hierarchical directory structure matching package organization.
"""

import importlib
import inspect
import os
from pathlib import Path
from typing import Any, List, Dict

from sphinx.application import Sphinx


def is_pydantic_model(obj: Any) -> bool:
    """Check if an object is a Pydantic BaseModel."""
    try:
        from pydantic import BaseModel
        return inspect.isclass(obj) and issubclass(obj, BaseModel) and obj is not BaseModel
    except (ImportError, TypeError):
        return False


def get_module_members(module_name: str):
    """Get all public members of a module."""
    try:
        module = importlib.import_module(module_name)
    except (ImportError, ModuleNotFoundError) as e:
        print(f"Warning: Could not import {module_name}: {e}")
        return [], [], []
    
    classes = []
    functions = []
    submodules = []
    
    for name in dir(module):
        if name.startswith("_"):
            continue
        
        try:
            obj = getattr(module, name)
            
            # Check if it's defined in this module (not imported)
            if hasattr(obj, "__module__") and obj.__module__ != module_name:
                continue
            
            if inspect.isclass(obj):
                classes.append((name, is_pydantic_model(obj)))
            elif inspect.isfunction(obj) or inspect.isbuiltin(obj):
                functions.append(name)
        except (AttributeError, ImportError):
            continue
    
    return classes, functions, submodules


def generate_module_rst(module_name: str, output_dir: Path, relative_to_base: str = "") -> str:
    """Generate RST file for a module in hierarchical structure."""
    classes, functions, _ = get_module_members(module_name)
    
    # Get just the module name (last part)
    module_parts = module_name.split(".")
    short_name = module_parts[-1]
    title = f"{short_name}"
    
    content = [
        title,
        "=" * len(title),
        "",
        f".. automodule:: {module_name}",
        "",
    ]
    
    # Add classes
    if classes:
        content.extend([
            "Classes",
            "-" * 40,
            "",
        ])
        
        for class_name, is_pydantic in classes:
            full_name = f"{module_name}.{class_name}"
            
            if is_pydantic:
                content.extend([
                    f".. autopydantic_model:: {full_name}",
                    "   :members:",
                    "   :undoc-members:",
                    "   :show-inheritance:",
                    "   :member-order: bysource",
                    "",
                ])
            else:
                content.extend([
                    f".. autoclass:: {full_name}",
                    "   :members:",
                    "   :undoc-members:",
                    "   :show-inheritance:",
                    "   :member-order: bysource",
                    "",
                ])
    
    # Add functions
    if functions:
        content.extend([
            "Functions",
            "-" * 40,
            "",
        ])
        
        for func_name in functions:
            full_name = f"{module_name}.{func_name}"
            content.extend([
                f".. autofunction:: {full_name}",
                "",
            ])
    
    # Write file
    output_file = output_dir / f"{short_name}.rst"
    output_file.write_text("\n".join(content))
    
    return short_name


def generate_package_index(package_name: str, output_dir: Path, submodules: List[str], subpackages: List[str]) -> None:
    """Generate index.rst for a package directory."""
    package_parts = package_name.split(".")
    short_name = package_parts[-1]
    title = f"{short_name}"
    
    content = [
        title,
        "=" * len(title),
        "",
        f".. automodule:: {package_name}",
        "",
    ]
    
    if submodules or subpackages:
        content.extend([
            ".. toctree::",
            "   :maxdepth: 1",
            "",
        ])
        
        # Add subpackages first
        for subpkg in sorted(subpackages):
            content.append(f"   {subpkg}/index")
        
        # Then add modules
        for mod in sorted(submodules):
            content.append(f"   {mod}")
        
        content.append("")
    
    index_file = output_dir / "index.rst"
    index_file.write_text("\n".join(content))


def generate_package_docs_hierarchical(app: Sphinx, package_name: str, output_base_dir: Path, parent_path: Path = None) -> Dict[str, Any]:
    """Recursively generate documentation for a package in hierarchical structure."""
    
    if parent_path is None:
        parent_path = output_base_dir
    
    try:
        package = importlib.import_module(package_name)
    except ImportError as e:
        print(f"Warning: Could not import package {package_name}: {e}")
        return {"modules": [], "subpackages": []}
    
    # Determine output directory for this package
    package_parts = package_name.split(".")
    
    # Create directory for this package
    if len(package_parts) > 1:
        # This is a subpackage, create directory under parent
        package_dir = parent_path / package_parts[-1]
    else:
        # Top-level package
        package_dir = parent_path / package_name
    
    package_dir.mkdir(parents=True, exist_ok=True)
    
    generated_modules = []
    generated_subpackages = []
    
    # Find submodules and subpackages
    if hasattr(package, "__path__"):
        package_path = Path(package.__path__[0])
        
        for item in package_path.iterdir():
            # Handle Python files (modules)
            if item.is_file() and item.suffix == ".py" and item.stem != "__init__":
                submodule_name = f"{package_name}.{item.stem}"
                try:
                    module_file = generate_module_rst(submodule_name, package_dir)
                    generated_modules.append(item.stem)
                except Exception as e:
                    print(f"Warning: Could not generate docs for {submodule_name}: {e}")
            
            # Handle subdirectories (subpackages)
            elif item.is_dir() and not item.name.startswith("_") and (item / "__init__.py").exists():
                subpackage_name = f"{package_name}.{item.name}"
                try:
                    result = generate_package_docs_hierarchical(app, subpackage_name, output_base_dir, package_dir)
                    generated_subpackages.append(item.name)
                except Exception as e:
                    print(f"Warning: Could not generate docs for {subpackage_name}: {e}")
    
    # Generate index.rst for this package
    generate_package_index(package_name, package_dir, generated_modules, generated_subpackages)
    
    return {
        "modules": generated_modules,
        "subpackages": generated_subpackages,
        "dir": package_dir
    }


def generate_main_index(app: Sphinx, packages: List[str], output_dir: Path, template_file: Path, package_info: Dict[str, Dict]):
    """Generate main index.rst using template or default, with expanded hierarchy."""
    
    if template_file and template_file.exists():
        # Use custom template - but add expanded hierarchy after it
        base_content = template_file.read_text()
        
        # Add detailed structure section
        hierarchy_section = [
            "",
            "",
            "Package Structure",
            "-" * 40,
            "",
            "Detailed view of all modules and subpackages:",
            "",
        ]
        
        for package in sorted(packages):
            if package in package_info:
                info = package_info[package]
                hierarchy_section.extend(generate_package_tree(package, info, output_dir, level=0))
        
        content = base_content + "\n".join(hierarchy_section)
    else:
        # Generate default index
        content = [
            "API Reference",
            "=" * 40,
            "",
            "Auto-generated API documentation with hierarchical organization.",
            "",
            ".. toctree::",
            "   :maxdepth: 2",
            "",
        ]
        
        for package in sorted(packages):
            content.append(f"   {package}/index")
        
        content = "\n".join(content)
    
    index_file = output_dir / "index.rst"
    index_file.write_text(content)


def generate_package_tree(package_name: str, info: Dict, base_dir: Path, level: int = 0) -> List[str]:
    """Generate a visual tree structure of a package showing its children."""
    indent = "   " * level
    package_parts = package_name.split(".")
    short_name = package_parts[-1]
    
    # Calculate relative path
    if level == 0:
        rel_path = f"{short_name}/index"
    else:
        # Build path from package name
        path_parts = package_name.split(".")[1:]  # Skip base
        rel_path = "/".join(path_parts) + "/index"
    
    lines = []
    
    # Add package entry with link
    if level == 0:
        lines.extend([
            f"**{short_name}** - :doc:`{short_name}/index`",
            "",
        ])
    else:
        lines.append(f"{indent}└── **{short_name}/** - :doc:`{short_name}/index`")
    
    # Add subpackages
    for subpkg_name in sorted(info.get('subpackages', [])):
        lines.append(f"{indent}    ├── **{subpkg_name}/** (subpackage)")
    
    # Add modules
    for mod_name in sorted(info.get('modules', [])):
        lines.append(f"{indent}    ├── {mod_name}.py")
    
    lines.append("")
    
    return lines


def setup_autogen(app: Sphinx):
    """Main setup function called by Sphinx."""
    
    # Get configuration
    packages = app.config.nemo_evaluator_packages
    if not packages:
        return
    
    # Determine output directory
    output_dir = Path(app.srcdir) / app.config.autogen_api_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Auto-generating hierarchical API docs in {output_dir}")
    
    # Generate docs for each package
    all_generated = {}
    for package in packages:
        print(f"  Generating docs for {package}...")
        try:
            result = generate_package_docs_hierarchical(app, package, output_dir)
            all_generated[package] = result
            print(f"    ✓ Generated: {len(result['modules'])} modules, {len(result['subpackages'])} subpackages")
        except Exception as e:
            print(f"    ✗ Error: {e}")
    
    # Generate main index with package info for hierarchy display
    template_file = Path(app.srcdir) / "_templates" / app.config.autogen_api_template
    generate_main_index(app, packages, output_dir, template_file, all_generated)
    
    total_packages = len(all_generated)
    total_modules = sum(len(info['modules']) for info in all_generated.values())
    total_subpackages = sum(len(info['subpackages']) for info in all_generated.values())
    
    print(f"Generated hierarchical docs: {total_packages} packages, {total_subpackages} subpackages, {total_modules} modules")


def setup(app: Sphinx):
    """Sphinx extension setup."""
    
    # Add configuration values
    app.add_config_value("nemo_evaluator_packages", [], "html")
    app.add_config_value("autogen_api_output_dir", "apidocs", "html")
    app.add_config_value("autogen_api_template", "autogen_api_index.rst", "html")
    
    # Connect to builder-inited event (runs before processing documents)
    app.connect("builder-inited", setup_autogen)
    
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
