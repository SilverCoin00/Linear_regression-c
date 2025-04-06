#include "Pandas_&_Numpy.c"
#include "Dataset.c"
#include <math.h>

typedef struct Standard_scaler {
    int features;
    float* mean;
    float* deviation;
} Standard_scaler;
typedef struct Min_max_scaler {
    int features;
    float* min;
    float* max;
} Min_max_scaler;

void scaler_fit(Dataset* ds, void* scaler, char* scaler_type) {
    if (!strcmp(scaler_type, "Standard_scaler")) {
        Standard_scaler* scl = (Standard_scaler*)scaler;
        scl->features = ds->features + 1;
        scl->mean = (float*)malloc(scl->features* sizeof(float));
        scl->deviation = (float*)malloc(scl->features* sizeof(float));
        float sum;
        int i, j;
        for (i = 0; i < scl->features - 1; i++) {
            sum = 0;
            for (j = 0; j < ds->samples; j++) sum += ds->x[j][i];
            scl->mean[i] = sum / ds->samples;
            sum = 0;
            for (j = 0; j < ds->samples; j++) sum += (scl->mean[i] - ds->x[j][i])* (scl->mean[i] - ds->x[j][i]);
            scl->deviation[i] = sqrt(sum / (ds->samples - 1));
        }
        for (i = 0, sum = 0; i < ds->samples; i++) sum += ds->y[i];
        scl->mean[scl->features - 1] = sum / ds->samples;
        for (i = 0, sum = 0; i < ds->samples; i++) 
            sum += (scl->mean[scl->features - 1] - ds->y[i])* (scl->mean[scl->features - 1] - ds->y[i]);
        scl->deviation[scl->features - 1] = sqrt(sum / (ds->samples - 1));
    } else if (!strcmp(scaler_type, "Min_max_scaler")) {
        Min_max_scaler* scl = (Min_max_scaler*)scaler;
        scl->features = ds->features + 1;
        scl->min = (float*)malloc(scl->features* sizeof(float));
        scl->max = (float*)malloc(scl->features* sizeof(float));
        int i, j;
        for (i = 0; i < scl->features - 1; i++) {
            scl->max[i] = scl->min[i] = ds->x[0][i];
            for (j = 1; j < ds->samples; j++) {
                if (scl->max[i] < ds->x[j][i]) scl->max[i] = ds->x[j][i];
                if (scl->min[i] > ds->x[j][i]) scl->min[i] = ds->x[j][i];
            }
        }
        scl->max[scl->features - 1] = scl->min[scl->features - 1] = ds->y[0];
        for (i = 1; i < ds->samples; i++) {
            if (scl->max[scl->features - 1] < ds->y[i]) scl->max[scl->features - 1] = ds->y[i];
            if (scl->min[scl->features - 1] > ds->y[i]) scl->min[scl->features - 1] = ds->y[i];
        }
    }
}
void scaler_transform(Dataset* ds, void* scaler, char* scaler_type) {
    if (!strcmp(scaler_type, "Standard_scaler")) {
        Standard_scaler* scl = (Standard_scaler*)scaler;
        int i, j;
        for (i = 0; i < ds->features; i++) {
            for (j = 0; j < ds->samples; j++) ds->x[j][i] = (ds->x[j][i] - scl->mean[i]) / scl->deviation[i];
        }
        for (i = 0; i < ds->samples; i++) 
            ds->y[i] = (ds->y[i] - scl->mean[scl->features - 1]) / scl->deviation[scl->features - 1];
    } else if (!strcmp(scaler_type, "Min_max_scaler")) {
        Min_max_scaler* scl = (Min_max_scaler*)scaler;
        int i, j;
        for (i = 0; i < ds->features; i++) {
            for (j = 0; j < ds->samples; j++) ds->x[j][i] = (ds->x[j][i] - scl->min[i]) / (scl->max[i] - scl->min[i]);
        }
        for (i = 0; i < ds->samples; i++) 
            ds->y[i] = (ds->y[i] - scl->min[scl->features - 1]) / (scl->max[scl->features - 1] - scl->min[scl->features - 1]);
    }
}
void free_scaler(void* scaler, char* scaler_type) {
    if (!strcmp(scaler_type, "Standard_scaler")) {
        Standard_scaler* scl = (Standard_scaler*)scaler;
        free(scl->mean);
        free(scl->deviation);
        free(scl);
    } else if (!strcmp(scaler_type, "Min_max_scaler")) {
        Min_max_scaler* scl = (Min_max_scaler*)scaler;
        free(scl->max);
        free(scl->min);
        free(scl);
    }
}
void train_test_split_ds(Dataset* data, Dataset* train, Dataset* test, float test_size, int random_state) {
    train->features = data->features;
    test->features = data->features;
    test->samples = round(test_size* data->samples);
    train->samples = data->samples - test->samples;
    test->x = (float**)malloc(test->samples* sizeof(float*));
    test->y = (float*)malloc(test->samples* sizeof(float));
    train->x = (float**)malloc(train->samples* sizeof(float*));
    train->y = (float*)malloc(train->samples* sizeof(float));

    int* random_i = (int*)malloc(data->samples* sizeof(int)), i;
    for (i = 0; i < data->samples; i++) random_i[i] = i;
    srand(random_state);
    for (int j = 0, t, a, b; j < data->samples / 2; j++) {
        a = rand() % data->samples, b = rand() % data->samples;
        t = random_i[a];
        random_i[a] = random_i[b];
        random_i[b] = t;
    }
    for (i = 0; i < test->samples; i++) {
        test->x[i] = (float*)malloc((test->features + 1)* sizeof(float));
        dataset_sample_copy(data, random_i[i], test, i);
    }
    for (int e = 0 ; i < data->samples; i++, e++) {
        train->x[e] = (float*)malloc((train->features + 1)* sizeof(float));
        dataset_sample_copy(data, random_i[i], train, e);
    }
    free(random_i);
}
