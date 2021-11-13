project('tjack', 'cpp', version : '0.1', default_options : ['cpp_std=c++20', 'buildtype=debugoptimized'])

cpp_compiler = meson.get_compiler('cpp')

# pthread
thread_dep = dependency('threads')

# libchess
libchess_src = [
    'libchess/chess/attack.cpp',
    'libchess/chess/board.cpp',
    'libchess/chess/chess.cpp',
    'libchess/chess/direction.cpp',
    'libchess/chess/game.cpp',
    'libchess/chess/move.cpp',
    'libchess/chess/piece.cpp',
    'libchess/chess/position.cpp',
    'libchess/chess/random.cpp',
    'libchess/chess/set.cpp',
    'libchess/chess/side.cpp',
    'libchess/chess/square.cpp',
    'libchess/chess/zobrist.cpp'
]

libchess_inc = include_directories('libchess')
libchess_dep = declare_dependency(sources : libchess_src, include_directories : libchess_inc)

# libtorch
py_mod = import('python')
py_installation = py_mod.find_installation('python3', modules : ['torch'])
py_purelib = py_installation.get_path('purelib')

pytorch_include_dir = py_purelib/'torch'/'include'
pytorch_api_include_dir = pytorch_include_dir/'torch'/'csrc'/'api'/'include'
pytorch_lib_dir = py_purelib/'torch'/'lib'

message('PyTorch include dir:', pytorch_include_dir)
message('PyTorch lib dir:', pytorch_lib_dir)

openmp_dep = dependency('openmp')

libtorch_dep = cpp_compiler.find_library('libtorch', dirs : pytorch_lib_dir)
libtorch_cpu_dep = cpp_compiler.find_library('libtorch_cpu', dirs : pytorch_lib_dir)
libtorch_cuda_dep = cpp_compiler.find_library('libtorch_cuda', dirs : pytorch_lib_dir, required : false)
libc10_dep = cpp_compiler.find_library('libc10', dirs : pytorch_lib_dir)
libc10_cuda_dep = cpp_compiler.find_library('libc10_cuda', dirs : pytorch_lib_dir, required : false)
libgomp_dep = cpp_compiler.find_library('libgomp', dirs : pytorch_lib_dir)

torch_inc = include_directories(run_command('scripts/rel_from_abs_path.py', pytorch_include_dir, meson.source_root()).stdout().strip(), run_command('scripts/rel_from_abs_path.py', pytorch_api_include_dir, meson.source_root()).stdout().strip())
torch_lib = [libtorch_dep, libtorch_cpu_dep, libc10_dep, libgomp_dep, openmp_dep]

has_cuda = run_command('scripts/has_cuda.py').stdout().strip() == '1'

if has_cuda
	message('PyTorch using CUDA')
	torch_lib += [libtorch_cuda_dep, libc10_cuda_dep]
else
	message('PyTorch not using CUDA')
endif

torch_dep = declare_dependency(
	include_directories : torch_inc,
	dependencies : torch_lib,
	compile_args : ['-D_GLIBCXX_USE_CXX11_ABI=@0@'.format(get_option('_GLIBCXX_USE_CXX11_ABI'))],
	link_args : ['-Wl,--no-as-needed']
)

# common
common_src = [
	'common/uci/uci.cpp',
	'common/output_thread.cpp'
]


common_inc = include_directories('common')

# example
example_src = [
	'example/main.cpp',
	'example/random_engine.cpp'
]

example = executable(
	'example',
	common_src + example_src,
	include_directories : [common_inc],
	dependencies : [libchess_dep, thread_dep, torch_dep]
)

# sigmazero
sigmazero_src = [
	'sigmazero/test_drive.cpp',
	'sigmazero/drl/sigmanet.cpp',
	'sigmazero/drl/action_encodings.cpp',
	'sigmazero/mcts/mcts.cpp',
	'sigmazero/mcts/node.cpp',
	'sigmazero/mcts/self_play.cpp'
]


sigmazero = executable(
	'sigmazero',
	common_src + sigmazero_src,
	include_directories : [common_inc],
	dependencies : [libchess_dep, thread_dep, torch_dep]
)

training = executable(
	'training',
	['sigmazero/training.cpp'],
	dependencies : [libchess_dep, torch_dep],
	cpp_pch : 'sigmazero/pch.hpp'
)

selfplay = executable(
	'selfplay',
	['sigmazero/selfplay.cpp'],
	dependencies : [libchess_dep, torch_dep],
	cpp_pch : 'sigmazero/pch.hpp'
)

# sigmazero
selfplay_src = [
	'sigmazero/selfplay_dummy.cpp',
	'sigmazero/mcts/misc.cpp',
	'sigmazero/mcts/node.cpp',
	'sigmazero/mcts/node.cpp',
	'sigmazero/mcts/self_play.cpp'
]

sigmanet = executable(
	'sigmanet',
	[
		'sigmazero/drl/sigmanet.cpp',
		'sigmazero/drl/sigma_test.cpp'
	],
	dependencies : [libchess_dep, torch_dep],
	cpp_pch : 'sigmazero/pch.hpp'
)
