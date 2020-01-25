project('png-parser', 'c', version: '0.1',
	default_options: ['c_std=c99', 'warning_level=3'])

zlib_dep = dependency('zlib')

executable('png_parser',
	'png_parser.c',
	dependencies: zlib_dep,
	install: true)
