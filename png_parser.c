#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <assert.h>

#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fcntl.h>
#include <unistd.h>

#include <zlib.h>

struct mapped_file {
	size_t size;
	uint8_t *ptr;
};

void map_file(const char *filename, struct mapped_file *out) {
	int fd = open(filename, O_RDONLY);
	if (fd < 0) {
		perror("failed to open file");
		exit(1);
	}

	struct stat st;
	if (fstat(fd, &st)) {
		perror("failed to stat file");
		exit(1);
	}

	size_t size = st.st_size;
	void *ptr = mmap(NULL, size, PROT_READ, MAP_PRIVATE, fd, 0);

	if (ptr == MAP_FAILED) {
		perror("failed to mmap file");
		exit(1);
	}

	if (close(fd)) {
		perror("failed to close file");
		exit(1);
	}

	if (out) {
		out->size = size;
		out->ptr = ptr;
	}
}

void unmap_file(struct mapped_file *file) {
	if (munmap(file->ptr, file->size) < 0) {
		perror("failed to unmap file");
		exit(1);
	}
}

struct png_chunk {
	uint32_t size;
	char type[4];
	void *data;
};

struct png_state {
	void *ptr;
	size_t size;
	size_t index;
};

#define ARR_SIZE(arr) (sizeof(arr) / (sizeof(*(arr))))

static inline __attribute__((always_inline)) void *offset_of(void *ptr, size_t off) {
	return (void *)((uintptr_t)ptr + off);
}

static inline __attribute__((always_inline)) uint32_t be_host32(uint32_t val) {
	uint8_t *buf = (uint8_t *)&val;
	uint32_t out = 0;

	out |= buf[3];
	out |= buf[2] << 8;
	out |= buf[1] << 16;
	out |= buf[0] << 24;

	return out;
}

static int png_fetchN(struct png_state *state, void *out, size_t count) {
	if (state->index + count > state->size)
		return 1;

	memcpy(out, offset_of(state->ptr, state->index), count);
	state->index += count;
	return 0;
}

static int png_fetch8(struct png_state *state, uint8_t *out) {
	return png_fetchN(state, out, 1);
}

static int png_fetch16(struct png_state *state, uint32_t *out) {
	return png_fetchN(state, out, 2);
}

static int png_fetch32(struct png_state *state, uint32_t *out) {
	return png_fetchN(state, out, 4);
}

static int png_fetch_next_chunk(struct png_state *state, struct png_chunk *out) {
	if (png_fetch32(state, &out->size))
		return 1;

	out->size = be_host32(out->size);

	if (png_fetchN(state, out->type, 4))
		return 1;

	if (state->index + out->size + 4 > state->size)
		return 1;

	out->data = offset_of(state->ptr, state->index);
	state->index += out->size + 4;

	return 0;
}

#define OUTPUT_SIZE_FACTOR 200

static inline __attribute__((always_inline)) int png_decompress_idat(struct png_state *state, void *out_data, size_t buf_size) {
	// duplicate the state to iterate over the png chunks without affecting the input state
	struct png_state state_copy = *state;
	void *input = NULL;
	size_t input_size = 0;

	struct png_chunk c;

	// gather scattered IDAT chunks into one buffer (I think this is correct)
	while(!png_fetch_next_chunk(&state_copy, &c)) {
		if (strncmp("IDAT", c.type, 4))
			continue;

		input = realloc(input, input_size + c.size);
		memcpy(offset_of(input, input_size), c.data, c.size);

		input_size += c.size;
	}

	// decompress
	size_t out_size = buf_size;
	int status = uncompress(out_data, &out_size, input, input_size);

	if (status == Z_MEM_ERROR) {
		fprintf(stderr, "failed to decompress: not enough memory\n");
		return 1;
	}

	if (status == Z_BUF_ERROR) {
		fprintf(stderr, "failed to decompress: not enough output buffer space\n");
		return 1;
	}

	if (status == Z_DATA_ERROR) {
		fprintf(stderr, "failed to decompress: broken data\n");
		return 1;
	}

	assert(status == Z_OK);
	assert(out_size == buf_size);

	return 0;
}

static int png_check(struct png_state *state) {
	if (state->size < 8)
		return 1;

	state->index += 8;

	return strncmp(offset_of(state->ptr, state->index - 8), "\x89PNG\r\n\x1A\n", 8);
}

static inline __attribute__((always_inline)) uint8_t png_sub_filter(uint8_t *line, size_t pixel_size, size_t x, size_t i) {
	if (x == 0)
		return line[pixel_size * x + i];

	return line[pixel_size * x + i] + line[pixel_size * (x - 1) + i];
}

static inline __attribute__((always_inline)) uint8_t png_up_filter(uint8_t *line, uint8_t *prev_line, size_t pixel_size, size_t x, size_t i) {
	if (!prev_line)
		return line[pixel_size * x + i];

	return line[pixel_size * x + i] + prev_line[pixel_size * x + i];
}


static inline __attribute__((always_inline)) uint8_t png_avg_filter(uint8_t *line, uint8_t *prev_line, size_t pixel_size, size_t x, size_t i) {
	uint8_t left = 0;
	uint8_t top = 0;

	if (x)
		left = line[pixel_size * (x - 1) + i];

	if (prev_line)
		top = prev_line[pixel_size * x + i];

	return ((left + top) / 2) + line[pixel_size * x + i];
}

static inline __attribute__((always_inline)) int png_abs(int x) {
	if (x < 0)
		return -x;

	return x;
}

static inline __attribute__((always_inline)) uint8_t png_paeth_filter(uint8_t *line, uint8_t *prev_line, size_t pixel_size, size_t x, size_t i) {
	// using int as calculations "must be performed exactly, without overflow"
	int a = 0, b = 0, c = 0, d, p;
	int pa, pb, pc;

	d = line[pixel_size * x + i];

	if (x)
		a = line[pixel_size * (x - 1) + i];

	if (prev_line)
		b = prev_line[pixel_size * x + i];

	if (prev_line && x)
		c = prev_line[pixel_size * (x - 1) + i];

	p = a + b - c;

	pa = png_abs(p - a);
	pb = png_abs(p - b);
	pc = png_abs(p - c);

	if (pa <= pb && pa <= pc)
		return d + a;
	if (pb <= pc)
		return d + b;

	return d + c;
}

int main(int argc, char **argv) {
	if(argc != 2) {
		printf("usage: %s filename\n", argv[0]);
		return 1;
	}

	struct mapped_file file;
	map_file(argv[1], &file);

	struct png_state state = {file.ptr, file.size, 0};

	// check header
	if (png_check(&state)) {
		fprintf(stderr, "not a png file\n");
		goto end;
	}

	struct png_chunk c;
	if (png_fetch_next_chunk(&state, &c))
		goto end;

	assert(!strncmp("IHDR", c.type, 4));
	assert(c.size == 13);
	uint32_t width = be_host32(*(uint32_t *)offset_of(c.data, 0));
	uint32_t height = be_host32(*(uint32_t *)offset_of(c.data, 4));
	uint8_t bit_depth = *(uint8_t *)offset_of(c.data, 8);
	uint8_t color_type = *(uint8_t *)offset_of(c.data, 9);
	uint8_t compression = *(uint8_t *)offset_of(c.data, 10);
	uint8_t filter = *(uint8_t *)offset_of(c.data, 11);
	uint8_t interlace = *(uint8_t *)offset_of(c.data, 12);

	printf("width: %u, height: %u, bpp: %hhu, color type: %hhu, compression: %hhu, filter: %hhu, interlace: %hhu\n",
			width, height, bit_depth, color_type, compression, filter, interlace);

	int is_truecolor = color_type & 0b010;
	int has_palette = color_type & 0b001;
	int has_alpha = color_type & 0b100;
	int is_grayscale = !is_truecolor;

	assert(is_truecolor && !has_palette);
	size_t pixel_size = (has_alpha ? 4 : 3) * (bit_depth / 8);

	assert(filter == 0);
	assert(interlace == 0); // TODO: add deinterlacing support

	size_t raw_size = (width * height * pixel_size) + height;
	void *raw_data = malloc(raw_size);

	if (png_decompress_idat(&state, raw_data, raw_size))
		goto end;

	printf("decompressed IDAT chunks, size %lu\n", raw_size);

	printf("writing PPM output\n");

	FILE *out = fopen("foo.ppm", "w");

	fprintf(out, "P3 %u %u 255\n", width, height);

	const char *filter_methods[] = {
		"none",
		"sub",
		"up",
		"average",
		"paeth"
	};

	uint8_t *buf = raw_data;
	for (size_t y = 0; y < height; y++) {
		uint8_t filter_method = buf[y * (width * pixel_size + 1)];
		printf("filter method for line: %s\n", filter_methods[filter_method]);

		uint8_t *prev_line = NULL;
		uint8_t *line = buf + y * (width * pixel_size + 1) + 1;

		if (y)
			prev_line = buf + (y - 1) * (width * pixel_size + 1) + 1;

		for (size_t x = 0; x < width; x++) {
			for (size_t i = 0; i < pixel_size; i++) {
				uint8_t actual_value;

				switch (filter_method) {
					case 0: // none
						actual_value = line[x * pixel_size + i];
						break;
					case 1: // sub
						actual_value = png_sub_filter(line, pixel_size, x, i);
						break;
					case 2: // up
						actual_value = png_up_filter(line, prev_line, pixel_size, x, i);
						break;
					case 3: // average
						actual_value = png_avg_filter(line, prev_line, pixel_size, x, i);
						break;
					case 4: // paeth
						actual_value = png_paeth_filter(line, prev_line, pixel_size, x, i);
						break;
					default:
						fprintf(stderr, "invalid filter %hhu\n", filter_method);
						actual_value = 0xFF;
				}

				line[x * pixel_size + i] = actual_value; // make filters actually work properly

				if (i < 3)
					fprintf(out, "%u ", actual_value);
			}
		}
		fprintf(out, "\n");
	}

	fclose(out);

	/*
	while(!png_fetch_next_chunk(&state, &c)) {
		const char *critical_chunks[] = {
			"IHDR",
			"PLTE",
			"IDAT",
			"IEND",
		};

		int is_critical = 0;
		for (size_t i = 0; i < ARR_SIZE(critical_chunks) && !is_critical; i++)
			is_critical = !strncmp(critical_chunks[i], c.type, 4);

		printf("found %s chunk \"%.4s\" of size %u bytes\n", is_critical ? "critical" : "ancillary", c.type, c.size);

		if (!is_critical)
			continue;

		if (!strncmp("IEND", c.type, 4))
			goto end;
	}*/

	// ...

end:
	unmap_file(&file);
}
