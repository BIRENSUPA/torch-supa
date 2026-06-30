# Copyright (C) 2020-2026 Shanghai Biren Technology Co., Ltd.

# Copyright © 2025 Shanghai Biren Technology Co., Ltd. All rights reserved.

"""cpp extension module support supa."""
import copy
import os
import shlex
import setuptools
import subprocess
import sys
import sysconfig
import warnings
import types
from distutils.errors import UnknownFileError
import torch
from torch.utils.file_baton import FileBaton
from torch.utils.hipify.hipify_python import GeneratedFileCleaner
from typing import List, Optional, Union, Tuple
from torch.torch_version import TorchVersion
from setuptools.command.build_ext import build_ext

from torch.utils.cpp_extension import (
    verify_ninja_availability,
    is_ninja_available,
    get_cxx_compiler,
    get_compiler_abi_compatibility_and_version,
    _run_ninja_build,
    _import_module_from_library,
    _get_exec_path,
    _nt_quote_args,
    get_default_build_root,
    _maybe_write,
    _check_and_build_extension_h_precompiler_headers,
    remove_extension_h_precompiler_headers,
    _get_pybind11_abi_build_flags,
    JIT_EXTENSION_VERSIONER,
    SUBPROCESS_DECODE_ARGS,
    IS_WINDOWS,
    LIB_EXT,
    EXEC_EXT,
    SHARED_FLAG,
    _TORCH_PATH,
    TORCH_LIB_PATH,
)

TORCH_SUPA_HOME = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

SUPA_RUNTIME_PATH = os.environ.get("SUPA_PATH", "/usr/local/birensupa/sdk/latest/supa")
FULLSTACK_HOME = os.path.realpath(f"{SUPA_RUNTIME_PATH}/../")

_SUDA_INCLUDE_PATH = os.getenv("SUDA_INCLUDE_PATH")
_SUDA_PLUGIN_HOME = os.getenv("SUDA_PLUGIN_HOME")

SUDA_CUDA_HOME = os.getenv("SUDA_CUDA_HOME")

__all__ = ["BuildExtension", "CppExtension", "SupaExtension", "load", "load_inline"]


def _get_supa_version() -> Optional[str]:
    with open(os.path.join(SUPA_RUNTIME_PATH, "version.txt")) as fi:
        return fi.read().strip()


def _check_build_with_cuda(extra_cflags: List[str]):
    try:
        import suda  # noqa: F401

        if extra_cflags is not None:
            for item in extra_cflags:
                if item.startswith("-I") and "/suda/" in item:
                    return True
                if item.startswith("-D") and "BUILD_WITH_CUDA" in item:
                    return True
        return False
    except ImportError:
        return False


def is_ccache_available() -> bool:
    """check existence of ccache. use ccache to accelerate process."""
    try:
        subprocess.check_output("ccache --version".split())
    except Exception:
        return False
    else:
        return True


# #############   Note for SUPA    ###############
# env setting scripts in fullstack set following environment variables:
# C_INCLUDE_PATH, CPLUS_INCLUDE_PATH, LIBRARY_PATH, LD_LIBRARY_PATH
# so that:
# 1. no need to add full path of library in ldflags
# 2. no need to add header path for cxx.
##################################################


COMMON_BRCC_FLAGS = [
    "-Werror",
    "-Wno-pass-failed",
    "-fPIC",
    "-fopenmp",
    "-Wno-deprecated-builtins",
    "-fdeclspec",
    "-Wno-pass-failed",
    "-Wno-absolute-value",
]

SUDA_FLAGS = [
    "-nosupawrapperinc",
    "-nogpuinc",
    "-fno-gpu-rdc",
    f"--supa-path={_SUDA_PLUGIN_HOME}/_supa",
    "-include",
    "supa.h",
    "-DENABLE_SUPA",
    "-D__CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__",
    "-D__CUDA_ARCH__=900",
    "-D__CUDA_ARCH_LIST__=900",
    "-D__CUDACC__",
    "-D__CUDA__",
    "-D__NVCC__=1",
    "-D__CUDACC_VER_MAJOR__=11",
    "-D__CUDACC_VER_MINOR__=8",
    "-D__CUDACC_VER_BUILD__=89",
    "-D__CUDA_API_VER_MAJOR__=11",
    "-D__CUDA_API_VER_MINOR__=8",
    "-D__NVCC_DIAG_PRAGMA_SUPPORT__=1",
]
COMMON_HOST_FLAGS = [
    "-fstack-protector-all",
    "-Wl,-z,relro,-z,now,-z,noexecstack",
    "-fPIE",
    "-pie",
    "-fPIC",
    "-Wno-terminate",
    "-Wno-error=terminate",
    "-fvisibility=hidden",
    "-Wno-narrowing",
    "-Wall",
    "-Wextra",
    "-Wno-missing-field-initializers",
    "-Wno-type-limits",
    "-Wno-array-bounds",
    "-Wno-unknown-pragmas",
    "-Wno-sign-compare",
    "-Wno-unused-parameter",
    "-Wno-unused-function",
    "-Wno-unused-result",
    "-Wno-strict-overflow",
    "-Wno-strict-aliasing",
    "-Wno-deprecated-declarations",
    "-Wno-ignored-qualifiers",
    "-Wno-write-strings",
    "-Wno-deprecated-copy",
    "-Wno-dangling-reference",
    "-Wno-stringop-overflow",
    "-Wno-error=pedantic",
    "-Wno-error=redundant-decls",
    "-Wno-error=old-style-cast",
    "-faligned-new",
    "-Werror",
    "-Wno-unused-but-set-variable",
    "-Wno-uninitialized",
    "-fno-math-errno",
    "-fno-trapping-math",
    "-finline-functions",
    "-fno-omit-frame-pointer",
    "-rdynamic",
]

SUPA_LIBRARIES = [
    "supa",
    "supart",
]

SUDA_LIBRARIES = [
    "cuda",
    "cudart",
]


def _check_supa_version(compiler_name: str, compiler_version: TorchVersion) -> None:
    """check supa runtime version. so far, nothing need."""
    pass


def _get_glibcxx_abi_build_flags():
    glibcxx_abi_cflags = ["-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI))]
    return glibcxx_abi_cflags


class BuildExtension(build_ext):
    """
    A custom :mod:`setuptools` build extension .

    This :class:`setuptools.build_ext` subclass takes care of passing the
    minimum required compiler flags (e.g. ``-std=c++17``) as well as mixed
    C++/SUPA compilation (and support for SUPA files in general).

    When using :class:`BuildExtension`, it is allowed to supply a dictionary
    for ``extra_compile_args`` (rather than the usual list) that maps from
    languages (``cxx`` or ``nvcc``) to a list of additional compiler flags to
    supply to the compiler. This makes it possible to supply different flags to
    the C++ and SUPA compiler during mixed compilation.

    ``use_ninja`` (bool): If ``use_ninja`` is ``True`` (default), then we
    attempt to build using the Ninja backend. Ninja greatly speeds up
    compilation compared to the standard ``setuptools.build_ext``.
    Fallbacks to the standard distutils backend if Ninja is not available.

    .. note::
        By default, the Ninja backend uses #CPUS + 2 workers to build the
        extension. This may use up too many resources on some systems. One
        can control the number of workers by setting the `MAX_JOBS` environment
        variable to a non-negative number.
    """

    @classmethod
    def with_options(cls, **options):
        """Return a subclass with alternative constructor that extends any original keyword arguments to the original constructor with the given options."""

        class cls_with_options(cls):  # type: ignore[misc, valid-type]
            def __init__(self, *args, **kwargs):
                kwargs.update(options)
                super().__init__(*args, **kwargs)

        return cls_with_options

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.no_python_abi_suffix = kwargs.get("no_python_abi_suffix", False)

        self.use_ninja = kwargs.get("use_ninja", True)
        if self.use_ninja:
            # Test if we can use ninja. Fallback otherwise.
            msg = (
                "Attempted to use ninja as the BuildExtension backend but "
                "{}. Falling back to using the slow distutils backend."
            )
            if not is_ninja_available():
                warnings.warn(msg.format("we could not find ninja."))
                self.use_ninja = False

    def finalize_options(self) -> None:
        super().finalize_options()
        if self.use_ninja:
            self.force = True

    def build_extensions(self) -> None:

        def object_filenames(self, source_filenames, strip_dir=0, output_dir=""):
            if output_dir is None:
                output_dir = ""
            obj_names = []
            for src_name in source_filenames:
                base, ext = os.path.splitext(src_name)
                base = os.path.splitdrive(base)[1]  # Chop off the drive
                base = base[os.path.isabs(base) :]  # If abs, chop off leading /
                if ext not in self.src_extensions:
                    raise UnknownFileError("unknown file type '%s' (from '%s')" % (ext, src_name))
                if strip_dir:
                    base = os.path.basename(base)
                # follow legacy naming convention xxx.su -> xxx.su.o
                if ext in (".su", ".cu"):
                    base = base + ext
                obj_names.append(os.path.join(output_dir, base + self.obj_extension))
            return obj_names

        self.compiler.object_filenames = types.MethodType(object_filenames, self.compiler)

        compiler_name, compiler_version = self._check_abi()

        supa_ext = False
        extension_iter = iter(self.extensions)
        extension = next(extension_iter, None)
        while not supa_ext and extension:
            for source in extension.sources:
                _, ext = os.path.splitext(source)
                if ext == ".su":
                    supa_ext = True
                    break
            extension = next(extension_iter, None)

        if supa_ext:
            _check_supa_version(compiler_name, compiler_version)

        for extension in self.extensions:
            # Ensure at least an empty list of flags for 'cxx' and 'brcc' when
            # extra_compile_args is a dict. Otherwise, default torch flags do
            # not get passed. Necessary when only one of 'cxx' and 'brcc' is
            # passed to extra_compile_args in SupaExtension, i.e.
            #   SupaExtension(..., extra_compile_args={'cxx': [...]})
            # or
            #   SupaExtension(..., extra_compile_args={'brcc': [...]})
            if isinstance(extension.extra_compile_args, dict):
                for ext in ["cxx", "brcc"]:
                    if ext not in extension.extra_compile_args:
                        extension.extra_compile_args[ext] = []

            self._add_compile_flag(extension, "-DTORCH_API_INCLUDE_EXTENSION_H")
            # See note [Pybind11 ABI constants]
            for name in ["COMPILER_TYPE", "STDLIB", "BUILD_ABI"]:
                val = getattr(torch._C, f"_PYBIND11_{name}", None)
                if val is not None and not IS_WINDOWS:
                    self._add_compile_flag(extension, f'-DPYBIND11_{name}="{val}"')
            self._define_torch_extension_name(extension)
            self._add_gnu_cpp_abi_flag(extension)

        if "brcc_dlink" in extension.extra_compile_args:
            assert self.use_ninja, f"With dlink=True, ninja is required to build supa extension {extension.name}."

        # Register .su, .suh, .hip, and .mm as valid source extensions.
        self.compiler.src_extensions += [".su", ".suh", ".cu", ".cuh"]
        # Save the original _compile method for later.
        if self.compiler.compiler_type == "msvc":
            self.compiler._cpp_extensions += [".su", ".suh"]
            original_compile = self.compiler.compile
            original_spawn = self.compiler.spawn
        else:
            original_compile = self.compiler._compile

        def append_std17_if_no_std_present(cflags) -> None:
            # brcc does not allow multiple -std to be passed, so we avoid
            # overriding the option if the user explicitly passed it.
            cpp_format_prefix = "/{}:" if self.compiler.compiler_type == "msvc" else "-{}="
            cpp_flag_prefix = cpp_format_prefix.format("std")
            cpp_flag = cpp_flag_prefix + "c++17"
            if not any(flag.startswith(cpp_flag_prefix) for flag in cflags):
                cflags.append(cpp_flag)

        def unix_supa_flags(cflags):
            cflags = COMMON_BRCC_FLAGS + cflags + _get_supa_arch_flags(cflags)
            return cflags

        def convert_to_absolute_paths_inplace(paths):
            # Helper function. See Note [Absolute include_dirs]
            if paths is not None:
                for i in range(len(paths)):
                    if not os.path.isabs(paths[i]):
                        paths[i] = os.path.abspath(paths[i])

        def unix_wrap_single_compile(obj, src, ext, cc_args, extra_postargs, pp_opts) -> None:
            # Copy before we make any modifications.
            cflags = copy.deepcopy(extra_postargs)
            try:
                original_compiler = self.compiler.compiler_so
                if _is_brcc_file(src):
                    brcc = [_join_supa_home("brcc", "bin", "brcc")]
                    self.compiler.set_executable("compiler_so", brcc)
                    if isinstance(cflags, dict):
                        cflags = cflags["brcc"]
                    cflags = unix_supa_flags(cflags)
                elif isinstance(cflags, dict):
                    cflags = cflags["cxx"]
                append_std17_if_no_std_present(cflags)

                original_compile(obj, src, ext, cc_args, cflags, pp_opts)
            finally:
                # Put the original compiler back in place.
                self.compiler.set_executable("compiler_so", original_compiler)

        def unix_wrap_ninja_compile(
            sources,
            output_dir=None,
            macros=None,
            include_dirs=None,
            debug=0,
            extra_preargs=None,
            extra_postargs=None,
            depends=None,
        ):
            r"""Compiles sources by outputting a ninja file and running it."""
            # NB: I copied some lines from self.compiler (which is an instance
            # of distutils.UnixCCompiler). See the following link.
            # https://github.com/python/cpython/blob/f03a8f8d5001963ad5b5b28dbd95497e9cc15596/Lib/distutils/ccompiler.py#L564-L567
            # This can be fragile, but a lot of other repos also do this
            # (see https://github.com/search?q=_setup_compile&type=Code)
            # so it is probably OK; we'll also get CI signal if/when
            # we update our python version (which is when distutils can be
            # upgraded)

            # Use absolute path for output_dir so that the object file paths
            # (`objects`) get generated with absolute paths.
            output_dir = os.path.abspath(output_dir)

            # See Note [Absolute include_dirs]
            convert_to_absolute_paths_inplace(self.compiler.include_dirs)
            _, objects, extra_postargs, pp_opts, _ = self.compiler._setup_compile(
                output_dir, macros, include_dirs, sources, depends, extra_postargs
            )
            common_cflags = self.compiler._get_cc_args(pp_opts, debug, extra_preargs)
            extra_cc_cflags = self.compiler.compiler_so[1:]
            with_supa = any(map(_is_brcc_file, sources))

            # extra_postargs can be either:
            # - a dict mapping cxx/brcc to extra flags
            # - a list of extra flags.
            if isinstance(extra_postargs, dict):
                post_cflags = extra_postargs["cxx"]
            else:
                post_cflags = list(extra_postargs)

            append_std17_if_no_std_present(post_cflags)

            supa_post_cflags = None
            supa_cflags = None
            supa_dlink_post_cflags = None
            suda_cflags = None
            suda_post_cflags = None

            if with_supa:
                supa_cflags = common_cflags
                if isinstance(extra_postargs, dict):
                    supa_post_cflags = extra_postargs["brcc"]
                else:
                    supa_post_cflags = list(extra_postargs)
                supa_post_cflags = unix_supa_flags(supa_post_cflags)
                append_std17_if_no_std_present(supa_post_cflags)
                supa_cflags = [shlex.quote(f) for f in supa_cflags]
                supa_post_cflags = [shlex.quote(f) for f in supa_post_cflags]

            if any(map(_is_suda_file, sources)):
                suda_cflags = copy.copy(supa_cflags)
                #  'Treat subsequent input files as supa'. must have for suda.
                suda_cflags.append("-xsupa")
                suda_post_cflags = copy.copy(supa_post_cflags)

            if isinstance(extra_postargs, dict) and "brcc_dlink" in extra_postargs:
                supa_dlink_post_cflags = unix_supa_flags(extra_postargs["brcc_dlink"])
                supa_dlink_post_cflags = [shlex.quote(f) for f in supa_dlink_post_cflags]

            _write_ninja_file_and_compile_objects(
                sources=sources,
                objects=objects,
                cflags=[shlex.quote(f) for f in extra_cc_cflags + common_cflags],
                post_cflags=[shlex.quote(f) for f in post_cflags],
                supa_cflags=supa_cflags,
                supa_post_cflags=supa_post_cflags,
                supa_dlink_post_cflags=supa_dlink_post_cflags,
                suda_cflags=suda_cflags,
                suda_post_cflags=suda_post_cflags,
                build_directory=output_dir,
                verbose=True,
                with_supa=with_supa,
            )

            # Return *all* object filenames, not just the ones we just built.
            return objects

        # Monkey-patch the _compile or compile method.
        # https://github.com/python/cpython/blob/dc0284ee8f7a270b6005467f26d8e5773d76e959/Lib/distutils/ccompiler.py#L511
        if self.use_ninja:
            self.compiler.compile = unix_wrap_ninja_compile
        else:
            self.compiler._compile = unix_wrap_single_compile

        build_ext.build_extensions(self)

    def get_ext_filename(self, ext_name):
        # Get the original shared library name. For Python 3, this name will be
        # suffixed with "<SOABI>.so", where <SOABI> will be something like
        # cpython-37m-x86_64-linux-gnu.
        ext_filename = super().get_ext_filename(ext_name)
        # If `no_python_abi_suffix` is `True`, we omit the Python 3 ABI
        # component. This makes building shared libraries with setuptools that
        # aren't Python modules nicer.
        if self.no_python_abi_suffix:
            # The parts will be e.g. ["my_extension", "cpython-37m-x86_64-linux-gnu", "so"].
            ext_filename_parts = ext_filename.split(".")
            # Omit the second to last element.
            without_abi = ext_filename_parts[:-2] + ext_filename_parts[-1:]
            ext_filename = ".".join(without_abi)
        return ext_filename

    def _check_abi(self) -> Tuple[str, TorchVersion]:
        # On some platforms, like Windows, compiler_cxx is not available.
        if hasattr(self.compiler, "compiler_cxx"):
            compiler = self.compiler.compiler_cxx[0]
        else:
            compiler = get_cxx_compiler()
        _, version = get_compiler_abi_compatibility_and_version(compiler)
        # Warn user if VC env is activated but `DISTUILS_USE_SDK` is not set.
        if IS_WINDOWS and "VSCMD_ARG_TGT_ARCH" in os.environ and "DISTUTILS_USE_SDK" not in os.environ:
            msg = (
                "It seems that the VC environment is activated but DISTUTILS_USE_SDK is not set."
                "This may lead to multiple activations of the VC env."
                "Please set `DISTUTILS_USE_SDK=1` and try again."
            )
            raise UserWarning(msg)
        return compiler, version

    def _add_compile_flag(self, extension, flag):
        extension.extra_compile_args = copy.deepcopy(extension.extra_compile_args)
        if isinstance(extension.extra_compile_args, dict):
            for args in extension.extra_compile_args.values():
                args.append(flag)
        else:
            extension.extra_compile_args.append(flag)

    def _define_torch_extension_name(self, extension):
        # pybind11 doesn't support dots in the names
        # so in order to support extensions in the packages
        # like torch._C, we take the last part of the string
        # as the library name
        names = extension.name.split(".")
        name = names[-1]
        define = f"-DTORCH_EXTENSION_NAME={name}"
        self._add_compile_flag(extension, define)

    def _add_gnu_cpp_abi_flag(self, extension):
        # use the same CXX ABI as what PyTorch was compiled with
        self._add_compile_flag(extension, "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI)))


def CppExtension(name, sources, *args, **kwargs):
    """
    Create a :class:`setuptools.Extension` for C++.

    Convenience method that creates a :class:`setuptools.Extension` with the
    bare minimum (but often sufficient) arguments to build a C++ extension.

    All arguments are forwarded to the :class:`setuptools.Extension`
    constructor. Full list arguments can be found at
    https://setuptools.pypa.io/en/latest/userguide/ext_modules.html#extension-api-reference

    .. note::
        The PyTorch python API (as provided in libtorch_python) cannot be built
        with the flag ``py_limited_api=True``.  When this flag is passed, it is
        the user's responsibility in their library to not use APIs from
        libtorch_python (in particular pytorch/python bindings) and to only use
        APIs from libtorch (aten objects, operators and the dispatcher). For
        example, to give access to custom ops from python, the library should
        register the ops through the dispatcher.

    Example:
        >>> # xdoctest: +SKIP
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CPP_EXT)
        >>> from setuptools import setup
        >>> from torch.utils.cpp_extension import BuildExtension, CppExtension
        >>> setup(
        ...     name='extension',
        ...     ext_modules=[
        ...         CppExtension(
        ...             name='extension',
        ...             sources=['extension.cpp'],
        ...             extra_compile_args=['-g'],
        ...             extra_link_args=['-Wl,--no-as-needed', '-lm'])
        ...     ],
        ...     cmdclass={
        ...         'build_ext': BuildExtension
        ...     })
    """

    include_dirs = copy.deepcopy(kwargs.get("include_dirs", []))
    include_dirs += include_paths()
    kwargs["include_dirs"] = include_dirs

    library_dirs = copy.deepcopy(kwargs.get("library_dirs", []))
    library_dirs += library_paths()
    kwargs["library_dirs"] = library_dirs

    libraries = copy.deepcopy(kwargs.get("libraries", []))
    libraries.append("c10")
    libraries.append("torch")
    libraries.append("torch_cpu")
    if not kwargs.get("py_limited_api", False):
        # torch_python uses more than the python limited api
        libraries.append("torch_python")

    kwargs["libraries"] = libraries

    extra_compile_args = copy.deepcopy(kwargs.get("extra_compile_args", []))
    extra_compile_args.extend(COMMON_HOST_FLAGS)

    kwargs["extra_compile_args"] = extra_compile_args
    kwargs["language"] = "c++"
    ret = setuptools.Extension(name, sources, *args, **kwargs)
    return ret


def _BrccExtension(kwargs):
    library_dirs = copy.deepcopy(kwargs.get("library_dirs", []))
    kwargs["library_dirs"] = library_dirs

    libraries = copy.deepcopy(kwargs.get("libraries", []))
    libraries.append("c10")
    libraries.append("torch")
    libraries.append("torch_cpu")
    if not kwargs.get("py_limited_api", False):
        # torch_python uses more than the python limited api
        libraries.append("torch_python")

    kwargs["libraries"] = libraries

    include_dirs = copy.deepcopy(kwargs.get("include_dirs", []))

    kwargs["include_dirs"] = include_dirs

    kwargs["language"] = "c++"

    extra_compile_args = copy.deepcopy(kwargs.get("extra_compile_args", {}))
    extra_compile_args["cxx"] = extra_compile_args.get("cxx", [])
    extra_compile_args["cxx"].extend(COMMON_HOST_FLAGS)
    extra_compile_args["brcc"] = extra_compile_args.get("brcc", [])
    extra_compile_args["brcc"].extend(COMMON_BRCC_FLAGS)

    kwargs["extra_compile_args"] = extra_compile_args


def SupaExtension(name, sources, *args, **kwargs):
    """
    Create a :class:`setuptools.Extension` for SUPA/C++.

    Convenience method that creates a :class:`setuptools.Extension` with the
    bare minimum (but often sufficient) arguments to build a SUPA/C++
    extension. This includes the SUPA include path, library path and runtime
    library.

    All arguments are forwarded to the :class:`setuptools.Extension`
    constructor. Full list arguments can be found at
    https://setuptools.pypa.io/en/latest/userguide/ext_modules.html#extension-api-reference

    .. note::
        The PyTorch python API (as provided in libtorch_python) cannot be built
        with the flag ``py_limited_api=True``.  When this flag is passed, it is
        the user's responsibility in their library to not use APIs from
        libtorch_python (in particular pytorch/python bindings) and to only use
        APIs from libtorch (aten objects, operators and the dispatcher). For
        example, to give access to custom ops from python, the library should
        register the ops through the dispatcher.

    Example:
        >>> # xdoctest: +SKIP
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CPP_EXT)
        >>> from setuptools import setup
        >>> from torch.utils.cpp_extension import BuildExtension, SupaExtension
        >>> setup(
        ...     name='supa_extension',
        ...     ext_modules=[
        ...         SupaExtension(
        ...                 name='supa_extension',
        ...                 sources=['extension.cpp', 'extension_kernel.su'],
        ...                 extra_compile_args={'cxx': ['-g'],
        ...                                     'brcc': ['-O2']},
        ...                 extra_link_args=['-Wl,--no-as-needed', '-lsupa'])
        ...     ],
        ...     cmdclass={
        ...         'build_ext': BuildExtension
        ...     })

    """
    _BrccExtension(kwargs)
    kwargs["library_dirs"].extend(library_paths(device_type="supa"))
    kwargs["libraries"].extend(library_names("supa"))
    kwargs["include_dirs"].extend(include_paths(device_type="supa"))
    return setuptools.Extension(name, sources, *args, **kwargs)


def SudaExtension(name, sources, *args, **kwargs):
    """
    Create a :class:`setuptools.Extension` for SUDA/C++.
    see help on SupaExtension except it accepts '.cu' file.
    """
    global _SUDA_INCLUDE_PATH
    if not _SUDA_INCLUDE_PATH:
        raise RuntimeError("can't find SUDA_INCLUDE_PATH. make sure suda is active.")
    _SUDA_INCLUDE_PATH = _SUDA_INCLUDE_PATH.split(":")
    # combine nvcc and brcc flags under brcc for supa extension.
    extra_compile_args = copy.deepcopy(kwargs.get("extra_compile_args", {}))
    brcc_args = extra_compile_args.get("brcc", [])
    brcc_args.extend(extra_compile_args.get("nvcc", []))
    extra_compile_args["brcc"] = brcc_args

    _BrccExtension(kwargs)

    kwargs["library_dirs"].extend(library_paths(device_type="suda"))
    kwargs["libraries"].extend(library_names("suda"))
    kwargs["include_dirs"].extend(include_paths(device_type="suda"))
    kwargs["extra_compile_args"]["brcc"].extend(SUDA_FLAGS)

    ret = setuptools.Extension(name, sources, *args, **kwargs)
    return ret


def include_paths(device_type: str = "cpu") -> List[str]:
    """
    Get the include paths required to build a C++ or SUPA extension.

    Args:
        device_type: Defaults to "cpu".
    Returns:
        A list of include path strings.
    """
    lib_include = os.path.join(_TORCH_PATH, "include")
    paths = [
        lib_include,
        # Remove this once torch/torch.h is officially no longer supported for C++ extensions.
        os.path.join(lib_include, "torch", "csrc", "api", "include"),
        # Some internal (old) Torch headers don't properly prefix their includes,
        # so we need to pass -Itorch/lib/include/TH as well.
        os.path.join(lib_include, "TH"),
        os.path.join(lib_include, "THC"),
    ]

    # add header files of torch_supa
    if TORCH_SUPA_HOME is not None:
        paths.append(TORCH_SUPA_HOME)
        paths.insert(0, os.path.join(TORCH_SUPA_HOME, "include"))
        paths.append(os.path.join(TORCH_SUPA_HOME, "include", "fmt", "include"))

    if device_type == "suda":
        paths.extend(_SUDA_INCLUDE_PATH)

    return paths


def library_names(device_type: str = "cpu") -> List[str]:
    if device_type == "supa":
        return SUPA_LIBRARIES
    elif device_type == "suda":
        return SUDA_LIBRARIES
    else:
        return []


def library_paths(device_type: str = "cpu") -> List[str]:
    """
    Get the library paths required to build a C++ or Supa extension.

    Args:
        device_type: Defaults to "cpu".

    Returns:
        A list of library path strings.
    """
    # We need to link against libtorch.so
    paths = [TORCH_LIB_PATH]
    paths.append(f"{TORCH_SUPA_HOME}/lib")
    if device_type == "supa":
        # nothing need.
        pass
    elif device_type == "suda":
        paths.append(f"{SUDA_CUDA_HOME}/lib")

    return paths


def load(
    name,
    sources: Union[str, List[str]],
    extra_cflags=None,
    extra_supa_cflags=None,
    extra_ldflags=None,
    extra_include_paths=None,
    build_directory=None,
    verbose=False,
    with_supa: Optional[bool] = None,
    is_python_module=True,
    is_standalone=False,
    keep_intermediates=True,
):
    """
    Load a PyTorch C++ extension just-in-time (JIT).

    To load an extension, a Ninja build file is emitted, which is used to
    compile the given sources into a dynamic library. This library is
    subsequently loaded into the current Python process as a module and
    returned from this function, ready for use.

    By default, the directory to which the build file is emitted and the
    resulting library compiled to is ``<tmp>/torch_extensions/<name>``, where
    ``<tmp>`` is the temporary folder on the current platform and ``<name>``
    the name of the extension. This location can be overridden in two ways.
    First, if the ``TORCH_EXTENSIONS_DIR`` environment variable is set, it
    replaces ``<tmp>/torch_extensions`` and all extensions will be compiled
    into subfolders of this directory. Second, if the ``build_directory``
    argument to this function is supplied, it overrides the entire path, i.e.
    the library will be compiled into that folder directly.

    To compile the sources, the default system compiler (``c++``) is used,
    which can be overridden by setting the ``CXX`` environment variable. To pass
    additional arguments to the compilation process, ``extra_cflags`` or
    ``extra_ldflags`` can be provided. For example, to compile your extension
    with optimizations, pass ``extra_cflags=['-O3']``. You can also use
    ``extra_cflags`` to pass further include directories.

    SUPA support with mixed compilation is provided. Simply pass SUPA source
    files (``.su`` or ``.suh``) along with other sources. Such files will be
    detected and compiled with brcc rather than the C++ compiler. This includes
    passing the SUPA lib64 directory as a library directory, and linking
    ``supa_runtime``. You can pass additional flags to brcc via
    ``extra_supa_cflags``, just like with ``extra_cflags`` for C++. Various
    heuristics for finding the SUPA install directory are used, which usually
    work fine. If not, setting the ``SUPA_HOME`` environment variable is the
    safest option.

    Args:
        name: The name of the extension to build. This MUST be the same as the
            name of the pybind11 module!
        sources: A list of relative or absolute paths to C++ source files.
        extra_cflags: optional list of compiler flags to forward to the build.
        extra_supa_cflags: optional list of compiler flags to forward to brcc
            when building SUPA sources.
        extra_ldflags: optional list of linker flags to forward to the build.
        extra_include_paths: optional list of include directories to forward
            to the build.
        build_directory: optional path to use as build workspace.
        verbose: If ``True``, turns on verbose logging of load steps.
        with_supa: Determines whether SUPA headers and libraries are added to
            the build. If set to ``None`` (default), this value is
            automatically determined based on the existence of ``.su`` or
            ``.suh`` in ``sources``. Set it to `True`` to force SUPA headers
            and libraries to be included.
        is_python_module: If ``True`` (default), imports the produced shared
            library as a Python module. If ``False``, behavior depends on
            ``is_standalone``.
        is_standalone: If ``False`` (default) loads the constructed extension
            into the process as a plain dynamic library. If ``True``, build a
            standalone executable.

    Returns:
        If ``is_python_module`` is ``True``:
            Returns the loaded PyTorch extension as a Python module.

        If ``is_python_module`` is ``False`` and ``is_standalone`` is ``False``:
            Returns nothing. (The shared library is loaded into the process as
            a side effect.)

        If ``is_standalone`` is ``True``.
            Return the path to the executable. (On Windows, TORCH_LIB_PATH is
            added to the PATH environment variable as a side effect.)

    Example:
        >>> # xdoctest: +SKIP
        >>> from torch.utils.cpp_extension import load
        >>> module = load(
        ...     name='extension',
        ...     sources=['extension.cpp', 'extension_kernel.su'],
        ...     extra_cflags=['-O2'],
        ...     verbose=True)
    """
    return _jit_compile(
        name,
        [sources] if isinstance(sources, str) else sources,
        extra_cflags,
        extra_supa_cflags,
        extra_ldflags,
        extra_include_paths,
        build_directory or _get_build_directory(name, verbose),
        verbose,
        with_supa,
        is_python_module,
        is_standalone,
        keep_intermediates=keep_intermediates,
    )


def load_inline(
    name,
    cpp_sources,
    supa_sources=None,
    functions=None,
    extra_cflags=None,
    extra_supa_cflags=None,
    extra_ldflags=None,
    extra_include_paths=None,
    build_directory=None,
    verbose=False,
    with_supa=None,
    is_python_module=True,
    with_pytorch_error_handling=True,
    keep_intermediates=True,
    use_pch=False,
):
    r'''
    Load a PyTorch C++ extension just-in-time (JIT) from string sources.

    This function behaves exactly like :func:`load`, but takes its sources as
    strings rather than filenames. These strings are stored to files in the
    build directory, after which the behavior of :func:`load_inline` is
    identical to :func:`load`.

    See `the
    tests <https://github.com/pytorch/pytorch/blob/master/test/test_cpp_extensions_jit.py>`_
    for good examples of using this function.

    Sources may omit two required parts of a typical non-inline C++ extension:
    the necessary header includes, as well as the (pybind11) binding code. More
    precisely, strings passed to ``cpp_sources`` are first concatenated into a
    single ``.cpp`` file. This file is then prepended with ``#include
    <torch/extension.h>``.

    Furthermore, if the ``functions`` argument is supplied, bindings will be
    automatically generated for each function specified. ``functions`` can
    either be a list of function names, or a dictionary mapping from function
    names to docstrings. If a list is given, the name of each function is used
    as its docstring.

    The sources in ``supa_sources`` are concatenated into a separate ``.su``
    file. The ``.cpp`` and ``.su`` files are compiled
    separately, but ultimately linked into a single library. Note that no
    bindings are generated for functions in ``supa_sources`` per se. To bind
    to a SUPA kernel, you must create a C++ function that calls it, and either
    declare or define this C++ function in one of the ``cpp_sources`` (and
    include its name in ``functions``).

    See :func:`load` for a description of arguments omitted below.

    Args:
        cpp_sources: A string, or list of strings, containing C++ source code.
        supa_sources: A string, or list of strings, containing SUPA source code.
        functions: A list of function names for which to generate function
            bindings. If a dictionary is given, it should map function names to
            docstrings (which are otherwise just the function names).
        with_supa: Determines whether SUPA headers and libraries are added to
            the build. If set to ``None`` (default), this value is
            automatically determined based on whether ``supa_sources`` is
            provided. Set it to ``True`` to force SUPA headers
            and libraries to be included.
        with_pytorch_error_handling: Determines whether pytorch error and
            warning macros are handled by pytorch instead of pybind. To do
            this, each function ``foo`` is called via an intermediary ``_safe_foo``
            function. This redirection might cause issues in obscure cases
            of cpp. This flag should be set to ``False`` when this redirect
            causes issues.

    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CPP_EXT)
        >>> from torch.utils.cpp_extension import load_inline
        >>> source = """
        at::Tensor sin_add(at::Tensor x, at::Tensor y) {
          return x.sin() + y.sin();
        }
        """
        >>> module = load_inline(name='inline_extension',
        ...                      cpp_sources=[source],
        ...                      functions=['sin_add'])

    .. note::
        Since load_inline will just-in-time compile the source code, please ensure
        that you have the right toolchains installed in the runtime. For example,
        when loading C++, make sure a C++ compiler is available. If you're loading
        a SUPA extension, you will need to additionally install the corresponding SUPA
        toolkit (brcc and any other dependencies your code has). Compiling toolchains
        are not included when you install torch and must be additionally installed.

        During compiling, by default, the Ninja backend uses #CPUS + 2 workers to build
        the extension. This may use up too many resources on some systems. One
        can control the number of workers by setting the `MAX_JOBS` environment
        variable to a non-negative number.
    '''
    build_directory = build_directory or _get_build_directory(name, verbose)

    if isinstance(cpp_sources, str):
        cpp_sources = [cpp_sources]
    supa_sources = supa_sources or []
    if isinstance(supa_sources, str):
        supa_sources = [supa_sources]

    cpp_sources.insert(0, "#include <torch/extension.h>")
    if not _check_build_with_cuda(extra_cflags):
        cpp_sources.insert(0, "#include <supa_runtime.h>")

    if use_pch is True:
        # Using PreCompile Header('torch/extension.h') to reduce compile time.
        _check_and_build_extension_h_precompiler_headers(extra_cflags, extra_include_paths)
    else:
        remove_extension_h_precompiler_headers()

    # If `functions` is supplied, we create the pybind11 bindings for the user.
    # Here, `functions` is (or becomes, after some processing) a map from
    # function names to function docstrings.
    if functions is not None:
        module_def = []
        module_def.append("PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {")
        if isinstance(functions, str):
            functions = [functions]
        if isinstance(functions, list):
            # Make the function docstring the same as the function name.
            functions = {f: f for f in functions}
        elif not isinstance(functions, dict):
            raise ValueError(f"Expected 'functions' to be a list or dict, but was {type(functions)}")
        for function_name, docstring in functions.items():
            if with_pytorch_error_handling:
                module_def.append(
                    f'm.def("{function_name}", torch::wrap_pybind_function({function_name}), "{docstring}");'
                )
            else:
                module_def.append(f'm.def("{function_name}", {function_name}, "{docstring}");')
        module_def.append("}")
        cpp_sources += module_def

    cpp_source_path = os.path.join(build_directory, "main.cpp")
    _maybe_write(cpp_source_path, "\n".join(cpp_sources))

    sources = [cpp_source_path]

    if supa_sources:
        supa_sources.insert(0, "#include <supa_runtime.h>")

        supa_source_path = os.path.join(build_directory, "kernel.su")
        _maybe_write(supa_source_path, "\n".join(supa_sources))

        sources.append(supa_source_path)

    return _jit_compile(
        name,
        sources,
        extra_cflags,
        extra_supa_cflags,
        extra_ldflags,
        extra_include_paths,
        build_directory,
        verbose,
        with_supa,
        is_python_module,
        is_standalone=False,
        keep_intermediates=keep_intermediates,
    )


def _jit_compile(
    name,
    sources,
    extra_cflags,
    extra_supa_cflags,
    extra_ldflags,
    extra_include_paths,
    build_directory: str,
    verbose: bool,
    with_supa: Optional[bool],
    is_python_module,
    is_standalone,
    keep_intermediates=True,
) -> None:
    if is_python_module and is_standalone:
        raise ValueError("`is_python_module` and `is_standalone` are mutually exclusive.")

    if with_supa is None:
        with_supa = any(map(_is_brcc_file, sources))

    old_version = JIT_EXTENSION_VERSIONER.get_version(name)

    args = {
        "name": name,
        "source_files": sources,
        "build_arguments": [extra_cflags, extra_supa_cflags, extra_ldflags, extra_include_paths],
        "build_directory": build_directory,
        "with_cuda": with_supa,
        "is_python_module": is_python_module,
        "is_standalone": is_standalone,
    }

    if TorchVersion(torch.__version__) >= (2, 7, 0):
        args["with_sycl"] = False

    version = JIT_EXTENSION_VERSIONER.bump_version_if_changed(**args)

    if version > 0:
        if version != old_version and verbose:
            print(
                f"The input conditions for extension module {name} have changed. "
                + f"Bumping to version {version} and re-building as {name}_v{version}...",
                file=sys.stderr,
            )
        name = f"{name}_v{version}"

    baton = FileBaton(os.path.join(build_directory, "lock"))
    if baton.try_acquire():
        try:
            if version != old_version:
                with GeneratedFileCleaner(keep_intermediates=keep_intermediates) as clean_ctx:
                    _write_ninja_file_and_build_library(
                        name=name,
                        sources=sources,
                        extra_cflags=extra_cflags or [],
                        extra_supa_cflags=extra_supa_cflags or [],
                        extra_ldflags=extra_ldflags or [],
                        extra_include_paths=extra_include_paths or [],
                        build_directory=build_directory,
                        verbose=verbose,
                        with_supa=with_supa,
                        is_standalone=is_standalone,
                    )
            elif verbose:
                print(
                    "No modifications detected for re-loaded extension " f"module {name}, skipping build step...",
                    file=sys.stderr,
                )
        finally:
            baton.release()
    else:
        baton.wait()

    if verbose:
        print(f"Loading extension module {name}...", file=sys.stderr)

    if is_standalone:
        return _get_exec_path(name, build_directory)

    return _import_module_from_library(name, build_directory, is_python_module)


def _write_ninja_file_and_compile_objects(
    sources: List[str],
    objects,
    cflags,
    post_cflags,
    supa_cflags,
    supa_post_cflags,
    supa_dlink_post_cflags,
    suda_cflags,
    suda_post_cflags,
    build_directory: str,
    verbose: bool,
    with_supa: Optional[bool],
) -> None:
    verify_ninja_availability()

    compiler = get_cxx_compiler()

    get_compiler_abi_compatibility_and_version(compiler)
    if with_supa is None:
        with_supa = any(map(_is_brcc_file, sources))
    build_file_path = os.path.join(build_directory, "build.ninja")
    if verbose:
        print(f"Emitting ninja build file {build_file_path}...", file=sys.stderr)

    # Create build_directory if it does not exist
    if not os.path.exists(build_directory):
        if verbose:
            print(f"Creating directory {build_directory}...", file=sys.stderr)
        # This is like mkdir -p, i.e. will also create parent directories.
        os.makedirs(build_directory, exist_ok=True)

    _write_ninja_file(
        path=build_file_path,
        cflags=cflags,
        post_cflags=post_cflags,
        supa_cflags=supa_cflags,
        supa_post_cflags=supa_post_cflags,
        supa_dlink_post_cflags=supa_dlink_post_cflags,
        suda_cflags=suda_cflags,
        suda_post_cflags=suda_post_cflags,
        sources=sources,
        objects=objects,
        ldflags=None,
        library_target=None,
        with_supa=with_supa,
    )
    if verbose:
        print("Compiling objects...", file=sys.stderr)
    _run_ninja_build(
        build_directory,
        verbose,
        # It would be better if we could tell users the name of the extension
        # that failed to build but there isn't a good way to get it here.
        error_prefix="Error compiling objects for extension",
    )


def _write_ninja_file_and_build_library(
    name,
    sources: List[str],
    extra_cflags,
    extra_supa_cflags,
    extra_ldflags,
    extra_include_paths,
    build_directory: str,
    verbose: bool,
    with_supa: Optional[bool],
    is_standalone: bool = False,
) -> None:
    verify_ninja_availability()

    compiler = get_cxx_compiler()

    get_compiler_abi_compatibility_and_version(compiler)
    if with_supa is None:
        with_supa = any(map(_is_supa_file, sources))
    extra_ldflags = _prepare_ldflags(extra_ldflags or [], with_supa, verbose, is_standalone)
    build_file_path = os.path.join(build_directory, "build.ninja")
    if verbose:
        print(f"Emitting ninja build file {build_file_path}...", file=sys.stderr)

    # Create build_directory if it does not exist
    if not os.path.exists(build_directory):
        if verbose:
            print(f"Creating directory {build_directory}...", file=sys.stderr)
        # This is like mkdir -p, i.e. will also create parent directories.
        os.makedirs(build_directory, exist_ok=True)

    if verbose:
        print(f"TORCH_SUPA_HOME is {TORCH_SUPA_HOME}...", file=sys.stderr)

    # NOTE: Emitting a new ninja build file does not cause re-compilation if
    # the sources did not change, so it's ok to re-emit (and it's fast).
    _write_ninja_file_to_build_library(
        path=build_file_path,
        name=name,
        sources=sources,
        extra_cflags=extra_cflags or [],
        extra_supa_cflags=extra_supa_cflags or [],
        extra_ldflags=extra_ldflags or [],
        extra_include_paths=extra_include_paths or [],
        with_supa=with_supa,
        is_standalone=is_standalone,
    )

    if verbose:
        print(f"Building extension module {name}...", file=sys.stderr)
    _run_ninja_build(build_directory, verbose, error_prefix=f"Error building extension '{name}'")


def _prepare_ldflags(extra_ldflags, with_supa, verbose, is_standalone):
    if IS_WINDOWS:
        print("Not support Windows platform", file=sys.stderr)
    else:
        extra_ldflags.append(f"-L{TORCH_LIB_PATH}")
        extra_ldflags.append("-lc10")
        extra_ldflags.append("-ltorch_cpu")
        extra_ldflags.append("-ltorch")

        extra_ldflags.extend([f"-l{x}" for x in SUPA_LIBRARIES])

        if not is_standalone:
            extra_ldflags.append("-ltorch_python")

        if is_standalone:
            extra_ldflags.append(f"-Wl,-rpath,{TORCH_LIB_PATH}")

    if with_supa:
        if verbose:
            print("Detected SUPA files, patching ldflags", file=sys.stderr)

        # seems no special ld flags for su files.
    return extra_ldflags


def _get_supa_arch_flags(cflags: Optional[List[str]] = None) -> List[str]:
    gpu_arch = os.getenv("DEVICE_ARCH", default="arch_20")
    return [f"--supa-gpu-arch={gpu_arch}"]


def _get_build_directory(name: str, verbose: bool) -> str:
    root_extensions_directory = os.environ.get("TORCH_EXTENSIONS_DIR")
    if root_extensions_directory is None:
        root_extensions_directory = get_default_build_root()
        supa_str = _get_supa_version().replace(".", "_")
        python_version = f'py{sys.version_info.major}{sys.version_info.minor}{getattr(sys, "abiflags", "")}'
        build_folder = f"{python_version}_supa_{supa_str}"

        root_extensions_directory = os.path.join(root_extensions_directory, build_folder)

    if verbose:
        print(f"Using {root_extensions_directory} as PyTorch extensions root...", file=sys.stderr)

    build_directory = os.path.join(root_extensions_directory, name)
    if not os.path.exists(build_directory):
        if verbose:
            print(f"Creating extension directory {build_directory}...", file=sys.stderr)
        # This is like mkdir -p, i.e. will also create parent directories.
        os.makedirs(build_directory, exist_ok=True)

    return build_directory


def _write_ninja_file_to_build_library(
    path, name, sources, extra_cflags, extra_supa_cflags, extra_ldflags, extra_include_paths, with_supa, is_standalone
) -> None:
    extra_cflags = [flag.strip() for flag in extra_cflags]
    extra_supa_cflags = [flag.strip() for flag in extra_supa_cflags]
    extra_ldflags = [flag.strip() for flag in extra_ldflags]
    extra_include_paths = [flag.strip() for flag in extra_include_paths]

    # Turn into absolute paths so we can emit them into the ninja build
    # file wherever it is.
    user_includes = [os.path.abspath(file) for file in extra_include_paths]

    # include_paths() gives us the location of torch/extension.h
    # TODO generalize with_supa as specific device type.
    if with_supa:
        system_includes = include_paths("supa")
    else:
        system_includes = include_paths("cpu")
    # sysconfig.get_path('include') gives us the location of Python.h
    # Explicitly specify 'posix_prefix' scheme on non-Windows platforms to workaround error on some MacOS
    # installations where default `get_path` points to non-existing `/Library/Python/M.m/include` folder
    python_include_path = sysconfig.get_path("include", scheme="nt" if IS_WINDOWS else "posix_prefix")
    if python_include_path is not None:
        system_includes.append(python_include_path)

    common_cflags = []
    if not is_standalone:
        common_cflags.append(f"-DTORCH_EXTENSION_NAME={name}")
        common_cflags.append("-DTORCH_API_INCLUDE_EXTENSION_H")

    # Windows does not understand `-isystem` and quotes flags later.
    if IS_WINDOWS:
        common_cflags += [f"-I{include}" for include in user_includes + system_includes]
    else:
        common_cflags += [f"-I{shlex.quote(include)}" for include in user_includes]
        common_cflags += [f"-isystem {shlex.quote(include)}" for include in system_includes]

    common_cflags += [f"{x}" for x in _get_glibcxx_abi_build_flags()]
    common_cflags += [f"{x}" for x in _get_pybind11_abi_build_flags()]

    if IS_WINDOWS:
        cflags = common_cflags + ["/std:c++17"] + extra_cflags
        cflags = _nt_quote_args(cflags)
    else:
        cflags = common_cflags + ["-fPIC", "-std=c++17"] + extra_cflags

    # add HOST flags
    cflags.extend(COMMON_HOST_FLAGS)

    if with_supa:
        supa_flags = common_cflags + COMMON_BRCC_FLAGS + _get_supa_arch_flags()
        supa_flags += extra_supa_cflags
        if not any(flag.startswith("-std=") for flag in supa_flags):
            supa_flags.append("-std=c++17")
        cc_env = os.getenv("CC")
        if cc_env is not None:
            supa_flags = ["-ccbin", cc_env] + supa_flags
    else:
        supa_flags = None

    def object_file_path(source_file: str) -> str:
        # '/path/to/file.cpp' -> 'file'
        file_name = os.path.splitext(os.path.basename(source_file))[0]
        if _is_supa_file(source_file) and with_supa:
            # Use a different object filename in case a C++ and SUPA file have
            # the same filename but different extension (.cpp vs. .su).
            target = f"{file_name}.su.o"
        else:
            target = f"{file_name}.o"
        return target

    objects = [object_file_path(src) for src in sources]
    ldflags = ([] if is_standalone else [SHARED_FLAG]) + extra_ldflags

    ext = EXEC_EXT if is_standalone else LIB_EXT
    library_target = f"{name}{ext}"

    _write_ninja_file(
        path=path,
        cflags=cflags,
        post_cflags=None,
        supa_cflags=supa_flags,
        supa_post_cflags=None,
        supa_dlink_post_cflags=None,
        suda_cflags=None,
        suda_post_cflags=None,
        sources=sources,
        objects=objects,
        ldflags=ldflags,
        library_target=library_target,
        with_supa=with_supa,
    )


def _write_ninja_file(
    path,
    cflags,
    post_cflags,
    supa_cflags,
    supa_post_cflags,
    supa_dlink_post_cflags,
    suda_cflags,
    suda_post_cflags,
    sources,
    objects,
    ldflags,
    library_target,
    with_supa,
) -> None:
    r"""Write a ninja file that does the desired compiling and linking.

    `path`: Where to write this file
    `cflags`: list of flags to pass to $cxx. Can be None.
    `post_cflags`: list of flags to append to the $cxx invocation. Can be None.
    `supa_cflags`: list of flags to pass to $brcc. Can be None.
    `supa_postflags`: list of flags to append to the $brcc invocation. Can be None.
    `sources`: list of paths to source files
    `objects`: list of desired paths to objects, one per source.
    `ldflags`: list of flags to pass to linker. Can be None.
    `library_target`: Name of the output library. Can be None; in that case,
                      we do no linking.
    `with_supa`: If we should be compiling with SUPA.
    """

    def sanitize_flags(flags):
        if flags is None:
            return []
        else:
            return [flag.strip() for flag in flags]

    cflags = sanitize_flags(cflags)
    post_cflags = sanitize_flags(post_cflags)
    supa_cflags = sanitize_flags(supa_cflags)
    supa_post_cflags = sanitize_flags(supa_post_cflags)
    supa_dlink_post_cflags = sanitize_flags(supa_dlink_post_cflags)
    ldflags = sanitize_flags(ldflags)

    # Sanity checks...
    assert len(sources) == len(objects)
    assert len(sources) > 0

    compiler = get_cxx_compiler()
    ccache = "ccache " if is_ccache_available() else ""

    # Version 1.3 is required for the `deps` directive.
    config = ["ninja_required_version = 1.3"]
    config.append(f"cxx = {ccache}{compiler}")

    with_suda = suda_cflags or suda_post_cflags

    if with_supa or supa_dlink_post_cflags:
        brcc = _join_supa_home("brcc", "bin", "brcc")
        config.append(f"brcc = {ccache}{brcc}")

    flags = [f'cflags = {" ".join(cflags)}']
    flags.append(f'post_cflags = {" ".join(post_cflags)}')
    if with_supa:
        flags.append(f'supa_cflags = {" ".join(supa_cflags)}')
        flags.append(f'supa_post_cflags = {" ".join(supa_post_cflags)}')
    if supa_dlink_post_cflags:
        flags.append(f'supa_dlink_post_cflags = {" ".join(supa_dlink_post_cflags)}')
    if with_suda:
        flags.append(f'suda_cflags = {" ".join(suda_cflags)}')
        flags.append(f'suda_post_cflags = {" ".join(suda_post_cflags)}')
    flags.append(f'ldflags = {" ".join(ldflags)}')

    # Turn into absolute paths so we can emit them into the ninja build
    # file wherever it is.
    sources = [os.path.abspath(file) for file in sources]

    # See https://ninja-build.org/build.ninja.html for reference.
    compile_rule = ["rule compile"]
    if IS_WINDOWS:
        compile_rule.append("  command = cl /showIncludes $cflags -c $in /Fo$out $post_cflags")
        compile_rule.append("  deps = msvc")
    else:
        compile_rule.append("  command = $cxx -MMD -MF $out.d $cflags -c $in -o $out $post_cflags")
        compile_rule.append("  depfile = $out.d")
        compile_rule.append("  deps = gcc")

    if with_supa:
        supa_compile_rule = ["rule supa_compile"]
        supa_compile_rule.append("  command = $brcc $supa_cflags -c $in -o $out $supa_post_cflags")

    if with_suda:
        suda_compile_rule = ["rule suda_compile"]
        # --generate-dependencies-with-compile is not supported by ROCm
        suda_compile_rule.append("  command = $brcc $suda_cflags -c $in -o $out $suda_post_cflags")

    # Emit one build rule per source to enable incremental build.
    build = []
    for source_file, object_file in zip(sources, objects):
        is_supa_source = _is_supa_file(source_file) and with_supa
        is_suda_source = _is_suda_file(source_file)
        rule = "supa_compile" if is_supa_source else ("suda_compile" if is_suda_source else "compile")
        if IS_WINDOWS:
            source_file = source_file.replace(":", "$:")
            object_file = object_file.replace(":", "$:")
        source_file = source_file.replace(" ", "$ ")
        object_file = object_file.replace(" ", "$ ")
        build.append(f"build {object_file}: {rule} {source_file}")

    if supa_dlink_post_cflags:
        supa_devlink_fb = os.path.join(os.path.dirname(objects[0]), "dlink.supafb")
        supa_devlink_s = os.path.join(os.path.dirname(objects[0]), "dlink.s")
        supa_devlink_output = os.path.join(os.path.dirname(objects[0]), "dlink.o")

        supa_devlink_rule = ["rule supa_devlink"]
        # Combine all steps in one rule:
        # 1. brcc --supa-link --supa-device-only -> .supafb
        # 2. Generate .s with .incbin for .supafb
        # 3. llvm-mc compile .s -> .o
        supa_devlink_rule.append(
            "  command = $brcc --supa-link --supa-device-only $in -o $supa_devlink_fb $supa_dlink_post_cflags && "
            'echo ".hidden __supa_fatbin" > $supa_devlink_s && '
            'echo ".type __supa_fatbin,@object" >> $supa_devlink_s && '
            'echo ".section .supa_fatbin,\\"a\\",@progbits" >> $supa_devlink_s && '
            'echo ".globl __supa_fatbin" >> $supa_devlink_s && '
            'echo ".p2align 16" >> $supa_devlink_s && '
            'echo "__supa_fatbin:" >> $supa_devlink_s && '
            'echo "  .incbin \\"$supa_devlink_fb\\"" >> $supa_devlink_s && '
            "llvm-mc -triple x86_64-unknown-linux-gnu -filetype=obj $supa_devlink_s -o $out"
        )
        supa_devlink = [f'build {supa_devlink_output}: supa_devlink {" ".join(objects)}']
        supa_devlink.append(f"  supa_devlink_fb = {supa_devlink_fb}")
        supa_devlink.append(f"  supa_devlink_s = {supa_devlink_s}")
        # Add the generated .o to objects for linking
        objects.append(supa_devlink_output)
    else:
        supa_devlink_rule, supa_devlink = [], []

    if library_target is not None:
        link_rule = ["rule link"]
        if IS_WINDOWS:
            cl_paths = subprocess.check_output(["where", "cl"]).decode(*SUBPROCESS_DECODE_ARGS).split("\r\n")
            if len(cl_paths) >= 1:
                cl_path = os.path.dirname(cl_paths[0]).replace(":", "$:")
            else:
                raise RuntimeError("MSVC is required to load C++ extensions")
            link_rule.append(f'  command = "{cl_path}/link.exe" $in /nologo $ldflags /out:$out')
        else:
            link_rule.append("  command = $cxx $in $ldflags -o $out")

        link = [f'build {library_target}: link {" ".join(objects)}']

        default = [f"default {library_target}"]
    else:
        link_rule, link, default = [], [], []

    # 'Blocks' should be separated by newlines, for visual benefit.
    blocks = [config, flags, compile_rule]
    if with_supa:
        blocks.append(supa_compile_rule)  # type: ignore[possibly-undefined]
    if with_suda:
        blocks.append(suda_compile_rule)

    blocks += [supa_devlink_rule, link_rule, build, supa_devlink, link, default]
    content = "\n\n".join("\n".join(b) for b in blocks)
    # Ninja requires a new lines at the end of the .ninja file
    content += "\n"
    _maybe_write(path, content)


def _join_supa_home(*paths) -> str:
    return os.path.join(FULLSTACK_HOME, *paths)


def _is_brcc_file(path: str) -> bool:
    return _is_supa_file(path) or _is_suda_file(path)


def _is_supa_file(path: str) -> bool:
    valid_ext = [".su", ".suh"]
    return os.path.splitext(path)[1] in valid_ext


def _is_suda_file(path: str) -> bool:
    valid_ext = [".cu", ".cuh"]
    return os.path.splitext(path)[1] in valid_ext
