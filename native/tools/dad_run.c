#include "distill_any_depth.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int parse_positive(const char* text, int* value) {
    char* end = NULL;
    const long parsed = strtol(text, &end, 10);
    if (end == text || *end != '\0' || parsed <= 0 || parsed > 2147483647L) {
        return 0;
    }
    *value = (int)parsed;
    return 1;
}

static int read_exact(const char* path, float* data, size_t count) {
    FILE* file = fopen(path, "rb");
    if (file == NULL) {
        fprintf(stderr, "cannot open input %s: %s\n", path, strerror(errno));
        return 0;
    }
    const size_t read = fread(data, sizeof(float), count, file);
    const int extra = fgetc(file);
    fclose(file);
    if (read != count || extra != EOF) {
        fprintf(
            stderr,
            "input must contain exactly %zu float32 values\n",
            count);
        return 0;
    }
    return 1;
}

static int write_exact(const char* path, const float* data, size_t count) {
    FILE* file = fopen(path, "wb");
    if (file == NULL) {
        fprintf(stderr, "cannot open output %s: %s\n", path, strerror(errno));
        return 0;
    }
    const size_t written = fwrite(data, sizeof(float), count, file);
    const int close_result = fclose(file);
    if (written != count || close_result != 0) {
        fprintf(stderr, "failed to write output %s\n", path);
        return 0;
    }
    return 1;
}

static int parse_encoder(const char* text, dad_encoder* encoder) {
    if (strcmp(text, "vits") == 0) {
        *encoder = DAD_ENCODER_VITS;
    } else if (strcmp(text, "vitb") == 0) {
        *encoder = DAD_ENCODER_VITB;
    } else if (strcmp(text, "vitl") == 0) {
        *encoder = DAD_ENCODER_VITL;
    } else {
        return 0;
    }
    return 1;
}

int main(int argc, char** argv) {
    if (argc != 7) {
        fprintf(
            stderr,
            "usage: %s MODEL {vits|vitb|vitl} WIDTH HEIGHT INPUT_F32 OUTPUT_F32\n",
            argv[0]);
        return 2;
    }

    int width = 0;
    int height = 0;
    dad_encoder encoder;
    if (!parse_encoder(argv[2], &encoder) ||
        !parse_positive(argv[3], &width) ||
        !parse_positive(argv[4], &height) ||
        width % 14 != 0 ||
        height % 14 != 0) {
        fprintf(stderr, "invalid encoder or dimensions\n");
        return 2;
    }

    const size_t pixels = (size_t)width * (size_t)height;
    if (pixels > SIZE_MAX / (3 * sizeof(float))) {
        fprintf(stderr, "dimensions overflow address space\n");
        return 2;
    }
    float* input = (float*)malloc(pixels * 3 * sizeof(float));
    float* output = (float*)malloc(pixels * sizeof(float));
    if (input == NULL || output == NULL) {
        fprintf(stderr, "out of memory\n");
        free(input);
        free(output);
        return 1;
    }
    if (!read_exact(argv[5], input, pixels * 3)) {
        free(input);
        free(output);
        return 1;
    }

    const dad_create_options options = {
        sizeof(dad_create_options),
        DAD_ABI_VERSION,
        encoder,
        0,
        0,
    };
    dad_context* context = NULL;
    dad_status status = dad_create(argv[1], &options, &context);
    if (status == DAD_STATUS_OK) {
        status = dad_infer_tensor_f32(
            context, input, width, height, output, pixels);
    }
    if (status != DAD_STATUS_OK) {
        fprintf(
            stderr,
            "%s: %s\n",
            dad_status_string(status),
            dad_last_error());
        dad_destroy(context);
        free(input);
        free(output);
        return 1;
    }

    const int written = write_exact(argv[6], output, pixels);
    dad_destroy(context);
    free(input);
    free(output);
    return written ? 0 : 1;
}
