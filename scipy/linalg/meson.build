# TODO: 64-bit BLAS and LAPACK

__init__py = fs.copyfile('__init__.py')
_cy_array_utils_pxd = fs.copyfile('_cythonized_array_utils.pxd')

_cython_tree += fs.copyfile('__init__.pxd')

_generate_pyx = find_program('_generate_pyx.py')
cython_linalg = custom_target('cython_linalg',
  output: [
    'cython_blas.pyx',
    'cython_lapack.pyx',
    'cython_blas.pxd',
    'cython_lapack.pxd',
    '_blas_subroutines.h',
    '_lapack_subroutines.h'
  ],
  input: '_generate_pyx.py',
  command: [_generate_pyx, '-o', '@OUTDIR@', accelerate_flag],
  depend_files: [
    'cython_blas_signatures.txt',
    'cython_lapack_signatures.txt',
  ],
  # TODO - we only want to install the .pxd files! See comments for
  #        `pxd_files` further down.
  install: true,
  install_dir: py3.get_install_dir() / 'scipy/linalg',
  install_tag: 'devel'
)
cython_blas_pxd = cython_linalg[2]
cython_lapack_pxd = cython_linalg[3]

# pyx -> c, pyx -> cpp generators, depending on __init__.py here.
linalg_init_cython_gen = generator(cython,
  arguments : cython_args,
  output : '@BASENAME@.c',
  depends : [_cython_tree, __init__py])

# pyx -> c, pyx -> cpp generators, depending on _cythonized_array_utils.pxd
linalg_init_utils_cython_gen = generator(cython,
  arguments : cython_args,
  output : '@BASENAME@.c',
  depends : [_cython_tree, __init__py, _cy_array_utils_pxd, cython_lapack_pxd])

# pyx -> c, pyx -> cpp generators, depending on copied pxd files and init
linalg_cython_gen = generator(cython,
  arguments : cython_args,
  output : '@BASENAME@.c',
  depends : [_cython_tree, __init__py, cython_blas_pxd])#, cython_linalg])

# fblas
fblas_module = custom_target('fblas_module',
  output: ['_fblasmodule.c', '_fblas-f2pywrappers.f'],
  input: 'fblas.pyf.src',
  command: [generate_f2pymod, '@INPUT@', '-o', '@OUTDIR@'] + f2py_freethreading_arg,
  depend_files:
    [
      'fblas_l1.pyf.src',
      'fblas_l2.pyf.src',
      'fblas_l3.pyf.src',
    ]
)

# Note: we're linking LAPACK on purpose here. For some routines (e.g., spmv)
# the float/double routines are in BLAS while the complex routines are in
# LAPACK - we have historically put these in `_fblas`.
py3.extension_module('_fblas',
  fblas_module,
  link_args: version_link_args,
  dependencies: [lapack_lp64_dep, fortranobject_dep],
  install: true,
  subdir: 'scipy/linalg'
)

# flapack
flapack_module = custom_target('flapack_module',
  output: ['_flapackmodule.c'],
  input: 'flapack.pyf.src',
  command: [generate_f2pymod, '@INPUT@', '-o', '@OUTDIR@'] + f2py_freethreading_arg,
  depend_files:
    [
      'flapack_user.pyf.src',
      'flapack_gen.pyf.src',
      'flapack_gen_banded.pyf.src',
      'flapack_gen_tri.pyf.src',
      'flapack_sym_herm.pyf.src',
      'flapack_pos_def.pyf.src',
      'flapack_pos_def_tri.pyf.src',
      'flapack_other.pyf.src',
    ]
)

# Note that -Wno-empty-body is Clang-specific and comes from `callstatement`s
# in flapack_other.pyf.src
py3.extension_module('_flapack',
  flapack_module,
  c_args: [Wno_empty_body],
  link_args: version_link_args,
  dependencies: [lapack_lp64_dep, fortranobject_dep],
  install: true,
  subdir: 'scipy/linalg'
)

# Add _fblas_64 and _flapack_64 if we're building with ILP64 support
#
# NOTE: what happened in the setup.py build was that we were linking LP64
# libopenblas.so to `_fblas` and ILP64 `libopenblas64_.so` to `_fblas_64`
# and used both at the same time. We never shipped wheels that way, it only
# worked in a CI job. We are re-exporting the LP64 symbols in
# `cython_blas`/`cython_lapack`, so we can't use only ILP64 even if we support
# it in all SciPy code.
# TODO: right now we're only detecting one BLAS library (like NumPy does), but
#       we need two blas and two lapack dependency objects here.
#       The ILP64 CI job in the 1.10.x branch downloads two OpenBLAS tarballs
#       and then uses both in the build (search for `Download-OpenBLAS('1'))
#       in azure-pipelines.yml if you want to check that).
if use_ilp64
  fblas64_module = custom_target('fblas64_module',
    output: ['_fblas_64module.c', '_fblas_64-f2pywrappers.f'],
    input: 'fblas_64.pyf.src',
    command: [generate_f2pymod, '@INPUT@', '-o', '@OUTDIR@'] + f2py_ilp64_opts + f2py_freethreading_arg,
    depend_files:
      [
        'fblas_l1.pyf.src',
        'fblas_l2.pyf.src',
        'fblas_l3.pyf.src',
      ]
  )

  py3.extension_module('_fblas_64',
    fblas64_module,
    link_args: version_link_args,
    include_directories: ['../_build_utils/src'],   # for npy_cblas.h
    dependencies: [lapack_dep, fortranobject_dep],  # lapack_dep is ILP64 if use_ilp64==true
    install: true,
    link_language: 'fortran',
    subdir: 'scipy/linalg'
  )

  flapack64_module = custom_target('flapack64_module',
    output: ['_flapack_64module.c', '_flapack_64-f2pywrappers.f'],
    input: 'flapack_64.pyf.src',
    command: [generate_f2pymod, '@INPUT@', '-o', '@OUTDIR@'] + f2py_ilp64_opts + f2py_freethreading_arg,
  )

  py3.extension_module('_flapack_64',
    flapack64_module,
    c_args: [Wno_empty_body],
    link_args: version_link_args,
    include_directories: ['../_build_utils/src'],   # for npy_cblas.h
    dependencies: [lapack_dep, fortranobject_dep],
    install: true,
    link_language: 'fortran',
    subdir: 'scipy/linalg'
  )
endif


# TODO: cblas/clapack are built *only* for ATLAS. Why? Is it still needed?

# _decomp_interpolative
py3.extension_module('_decomp_interpolative',
  linalg_cython_gen.process('_decomp_interpolative.pyx'),
  c_args: cython_c_args,
  dependencies: np_dep,
  link_args: version_link_args,
  install: true,
  subdir: 'scipy/linalg'
)

# batched linalg inv module
py3.extension_module('_batched_linalg',
  [
    'src/_common_array_utils.hh',
    'src/_linalg_inv.hh',
    'src/_linalg_solve.hh',
    'src/_npymath.hh',
    'src/_batched_linalg_module.cc'
  ],
  dependencies: [np_dep, lapack_dep],
  include_directories: ['../_build_utils/src'],
  install: true,
  subdir: 'scipy/linalg'
)

# _solve_toeplitz
py3.extension_module('_solve_toeplitz',
  linalg_init_cython_gen.process('_solve_toeplitz.pyx'),
  c_args: cython_c_args,
  dependencies: np_dep,
  link_args: version_link_args,
  install: true,
  subdir: 'scipy/linalg'
)

# _matfuncs_sqrtm_triu:
py3.extension_module('_matfuncs_sqrtm_triu',
  linalg_init_cython_gen.process('_matfuncs_sqrtm_triu.pyx'),
  c_args: cython_c_args,
  dependencies: np_dep,
  link_args: version_link_args,
  install: true,
  subdir: 'scipy/linalg'
)

cython_blas = py3.extension_module('cython_blas',
  [
    linalg_cython_gen.process(cython_linalg[0]),  # cython_blas.pyx
    cython_linalg[4],  # _blas_subroutines.h
  ],
  c_args: cython_c_args,
  link_args: version_link_args,
  dependencies: [lapack_lp64_dep, np_dep],
  install: true,
  include_directories: ['../_build_utils/src'],
  subdir: 'scipy/linalg'
)

cython_lapack = py3.extension_module('cython_lapack',
  [
    linalg_cython_gen.process(cython_linalg[1]),  # cython_lapack.pyx
    cython_linalg[5],  # _lapack_subroutines.h
  ],
  c_args: cython_c_args,
  link_args: version_link_args,
  dependencies: [lapack_lp64_dep, np_dep],
  install: true,
  include_directories: ['../_build_utils/src'],
  subdir: 'scipy/linalg'
)

py3.extension_module('_decomp_lu_cython',
  linalg_cython_gen.process('_decomp_lu_cython.pyx'),
  c_args: cython_c_args,
  dependencies: np_dep,
  link_args: version_link_args,
  install: true,
  subdir: 'scipy/linalg'
)

_decomp_update_pyx = custom_target('_decomp_update',
  output: '_decomp_update.pyx',
  input: '_decomp_update.pyx.in',
  command: [tempita, '@INPUT@', '-o', '@OUTDIR@']
)

py3.extension_module('_decomp_update',
  linalg_cython_gen.process(_decomp_update_pyx),
  c_args: cython_c_args,
  dependencies: np_dep,
  link_args: version_link_args,
  install: true,
  subdir: 'scipy/linalg'
)

py3.extension_module('_matfuncs_expm',
  [
    '_matfuncs_expm.h',
    '_matfuncs_expm.c',
  ],
  dependencies: [np_dep, lapack_lp64_dep],
  install: true,
  subdir: 'scipy/linalg'
)

py3.extension_module('_matfuncs_schur_sqrtm',
  [
    '_common_array_utils.h',
    '_matfuncs_sqrtm.h',
    '_matfuncs_sqrtm.c',
  ],
  dependencies: [np_dep, lapack_lp64_dep],
  install: true,
  subdir: 'scipy/linalg'
)

_cythonized_array_utils = py3.extension_module('_cythonized_array_utils',
  linalg_init_utils_cython_gen.process('_cythonized_array_utils.pyx'),
  c_args: [cython_c_args, Wno_maybe_uninitialized],
  dependencies: np_dep,
  link_args: version_link_args,
  install: true,
  subdir: 'scipy/linalg'
)

if use_pythran
  py3.extension_module('_linalg_pythran',
    pythran_gen.process('_linalg_pythran.py'),
    cpp_args: cpp_args_pythran,
    dependencies: [pythran_dep, np_dep],
    link_args: version_link_args,
    install: true,
    subdir: 'scipy/linalg'
  )
else
  py3.install_sources(
    ['_linalg_pythran.py'],
    subdir: 'scipy/linalg'
  )
endif


python_sources = [
  '__init__.py',
  '__init__.pxd',
  '_basic.py',
  '_cythonized_array_utils.pxd',
  '_cythonized_array_utils.pyi',
  '_decomp.py',
  '_decomp_cholesky.py',
  '_decomp_cossin.py',
  '_decomp_ldl.py',
  '_decomp_lu.py',
  '_decomp_lu_cython.pyi',
  '_decomp_polar.py',
  '_decomp_qr.py',
  '_decomp_qz.py',
  '_decomp_schur.py',
  '_decomp_svd.py',
  '_expm_frechet.py',
  '_matfuncs.py',
  '_matfuncs_expm.pyi',
  '_matfuncs_inv_ssq.py',
  '_matfuncs_sqrtm.py',
  '_misc.py',
  '_procrustes.py',
  '_sketches.py',
  '_solvers.py',
  '_special_matrices.py',
  '_testutils.py',
  'basic.py',
  'blas.py',
  'decomp.py',
  'decomp_cholesky.py',
  'decomp_lu.py',
  'decomp_qr.py',
  'decomp_schur.py',
  'decomp_svd.py',
  'interpolative.py',
  'lapack.py',
  'matfuncs.py',
  'misc.py',
  'special_matrices.py'
]

py3.install_sources(
  python_sources,
  subdir: 'scipy/linalg'
)



# TODO: install (only) these pxd files. Neither py3.install_sources nor
#       install_data accept generated data files. What
#       https://mesonbuild.com/Installing.html says is for build targets to
#       use:
#         `custom_target(..., install: true, install_dir: ...)
#         # should use `py3.get_install_dir() / 'scipy/linalg'` ?
#       see https://github.com/mesonbuild/meson/issues/3206
#
#       For the below code to work, the script generating the files should use
#       a different filename, and then it should be moved to the final location
#       (e.g. with `fs.copyfile`). Either that, or split the codegen scripts and
#       call it twice: once for the installable files, and once for the
#       non-installable files.
#
#pxd_files = [
#  cython_linalg[2],  # cython_blas.pxd
#  cython_linalg[3],  # cython_lapack.pxd
#]
#py.install_sources(pxd_files)

#custom_target('cython_blas_lapack_pxd',
#  input : pxd_files,
#  output : ['cython_blas2.pxd', 'cython_lapack2.pxd'],
#  command : ['cp', '@INPUT0@', '@OUTPUT0@', '&&', 'cp', '@INPUT1@', '@OUTPUT1@'],
#  install : true,
#  install_dir: py3.get_install_dir() / 'scipy/linalg'
#)

subdir('tests')
