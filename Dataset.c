#include "Core.h"

typedef struct Dataset {
	float** x;
	float* y;
	int features;      // x_cols
	int samples;       // rows
} Dataset;

Dataset* new_dataset(float** x_train, float* y_train, int num_of_features, int num_of_samples) {
	Dataset* newd = (Dataset*)malloc(sizeof(Dataset));
	newd->x = (float**)malloc(num_of_samples* sizeof(float*));
	for (int i = 0; i < num_of_samples; i++) {
		newd->x[i] = (float*)malloc((num_of_features + 1)* sizeof(float));
		for (int j = 0; j < num_of_features; j++) newd->x[i][j] = x_train[i][j];
		newd->x[i][num_of_features] = 1;                                             // spare 1 col for bias
	}
	newd->y = (float*)malloc(num_of_samples* sizeof(float));
	for (int i = 0; i < num_of_samples; i++) newd->y[i] = y_train[i];
	newd->features = num_of_features;
	newd->samples = num_of_samples;
	return newd;
}
Dataset* trans_dframe_to_dset(Data_Frame* df, const char* predict_feature_col) {
	int y_col = strtoi(predict_feature_col), i, j, k;
	if (y_col < 0) {
		for (i = 0; i < df->col; i++) {
			if (strcmp(df->features[i], predict_feature_col) == 0) {
				y_col = i;
				break;
			}
		}
	}
	Dataset* newd = (Dataset*)malloc(sizeof(Dataset));
	newd->x = (float**)malloc(df->row* sizeof(float*));
	for (i = 0; i < df->row; i++) {
		newd->x[i] = (float*)malloc(df->col* sizeof(float));         // drop 1 col for y but plus 1 for bias, so, nothing changes
		for (j = 0, k = 0; j < df->col; k++) {
			if (k == y_col) continue;
			newd->x[i][j++] = df->data[i][k];
		}
		newd->x[i][df->col - 1] = 1;
	}
	newd->y = (float*)calloc(df->row, sizeof(float));
	if (y_col >= 0) for (i = 0; i < df->row; i++) newd->y[i] = df->data[i][y_col];;
	newd->features = df->col - 1;
	newd->samples = df->row;
	return newd;
}
Dataset* dataset_copy(const Dataset* ds) {
	Dataset* newd = (Dataset*)malloc(sizeof(Dataset));
	newd->x = (float**)malloc(ds->samples* sizeof(float*));
	for (int i = 0; i < ds->samples; i++) {
		newd->x[i] = (float*)malloc((ds->features + 1)* sizeof(float));
		for (int j = 0; j < ds->features; j++) newd->x[i][j] = ds->x[i][j];
		newd->x[i][ds->features] = 1;
	}
	newd->y = (float*)malloc(ds->samples* sizeof(float));
	for (int i = 0; i < ds->samples; i++) newd->y[i] = ds->y[i];
	newd->features = ds->features;
	newd->samples = ds->samples;
	return newd;
}
void dataset_sample_copy(const Dataset* ds, int ds_sample_index, Dataset* copy, int copy_sample_index) {
	if (!copy->x[copy_sample_index]) copy->x[copy_sample_index] = (float*)malloc((copy->features + 1)* sizeof(float));
	for (int i = 0; i < ds->features && i < copy->features; i++)
		copy->x[copy_sample_index][i] = ds->x[ds_sample_index][i];
	copy->x[copy_sample_index][copy->features] = 1;
	copy->y[copy_sample_index] = ds->y[ds_sample_index];
}
void print_dataset(Dataset* ds, int decimal, int col_space, int num_of_rows) {
	if (!ds) return ;
	if (num_of_rows < 0 || num_of_rows > ds->samples) num_of_rows = ds->samples;
	printf(" Row\n");
	for (int i = 0, j; i < num_of_rows; i++) {
		printf("%4d\t", i);
		for (j = 0; j < ds->features; j++) {
			printf("%*.*f ", col_space, decimal, ds->x[i][j]);
		}
		printf("\t| %*.*f\n", col_space, decimal, ds->y[i]);
	}
}
void free_dataset(Dataset* ds) {
	for (int i = 0; i < ds->samples; i++) free(ds->x[i]);
	free(ds->x);
	free(ds->y);
	free(ds);
}
