#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Pandas:

typedef struct Data_Frame {
    char** features;
    float** data;
    int row, col;
} Data_Frame;

static int is_blank(char* s) {
  	for (int i = 0; s[i]; i++) if (s[i] != ' ' && s[i] != '\t' && s[i] != '\n' && s[i] != '\r') return 0;;
  	return 1;
}
static int count_row(FILE* file, int max_line_length) {
  	char* s = (char*)malloc(max_line_length* sizeof(char));
  	int count;
  	for (count = 0; fgets(s, max_line_length, file); ) if (!is_blank(s)) count++;;
  	free(s);
  	rewind(file);
  	return count;
}
static int count_col(FILE* file, int max_line_length, char* seperate) {
  	char* s = (char*)malloc(max_line_length* sizeof(char));
  	if (fgets(s, max_line_length, file) == NULL) {
    		free(s);
    		return 0;
  	}
  	int count = 1;
  	for (int i = 0; s[i] != '\n'; i++) {
  		if (s[i] == seperate[0]) count++;
  	}
  	free(s);
  	rewind(file);
  	return count;
}
int strtoi(const char* number) {
  	int num = 0;
  	for (int i = 0; number[i]; i++) {
    		if (number[i] < 48 || number[i] > 57) return -1;
    		num *= 10;
    		num += number[i] - 48;
  	}
  	return num;
}
Data_Frame* read_csv(char* file_name, int max_line_length, char* seperate) {
    FILE* file = fopen(file_name, "r");
  	if (!file) {
    		printf("Error: Cannot open file !!");
    		return NULL;
  	}
    Data_Frame* newd = (Data_Frame*)malloc(sizeof(Data_Frame));
  	newd->row = count_row(file, max_line_length) - 1;
    newd->col = count_col(file, max_line_length, seperate);
  	newd->features = (char**)malloc(newd->col* sizeof(char*));
  	newd->data = (float**)malloc(newd->row* sizeof(float*));
  
    char* s = (char*)malloc(max_line_length* sizeof(char));
  	char* token;
    fgets(s, max_line_length, file);
    if (s[0] < 48 || s[0] > 57) {
		token = strtok(s, seperate);
		for (int i = 0, size; token != NULL && i < newd->col; i++) {
			size = strlen(token) + 1;
			newd->features[i] = (char*)malloc(size* sizeof(char));
			snprintf(newd->features[i], size, "%s", token);
			token = strtok(NULL, seperate);
		  }
		if (newd->features[newd->col - 1]) 
			  newd->features[newd->col - 1][strcspn(newd->features[newd->col - 1], "\n")] = '\0';
	} else rewind(file);
  
  	float num;
  	for (int i = 0, j = 0; i < newd->row; i++, j = 0) {
    		newd->data[i] = (float*)malloc(newd->col* sizeof(float));
        fgets(s, max_line_length, file);
    		token = strtok(s, seperate);
    		while (token != NULL && j < newd->col) {
      			sscanf(token, "%f", &num);
      			newd->data[i][j++] = num;
      			token = strtok(NULL, seperate);
    		}
  	}
    free(s);
    fclose(file);
    return newd;
}
void make_csv(char* file_name_csv, Data_Frame* df, char* seperate) {
    FILE* file = fopen(file_name_csv, "w");
  	if (!file) {
    		printf("Error: Cannot open file !!");
    		return ;
  	}
  	if (!df) return ;
  	int i, j;
  	if (df->features) {
    		for (i = 0; i < df->col; i++) 
    			  fprintf(file, "%s%s", df->features[i] ? df->features[i] : "null", i == df->col - 1 ? "\n" : seperate);
  	}
  	for (i = 0; i < df->row; i++) {
  		  for (j = 0; j < df->col; j++) {
  			    fprintf(file, "%f%s", df->data[i][j], j == df->col - 1 ? "\n" : seperate);
  		  }
  	}
  	fclose(file);
}
void print_data_frame(Data_Frame* df, int col_space, int num_of_rows) {
    if (!df) {
        printf("Error: DataFrame is not existing !!");
        return ;
    }
	  if (num_of_rows < 0 || num_of_rows > df->row) num_of_rows = df->row;
    int i, j;
    printf("\t");
    for (i = 0; i < df->col; i++) printf("%*s ", col_space, df->features[i]);
    printf("\n");
    for (i = 0; i < num_of_rows; i++) {
        printf("%5d\t", i + 1);
        for (j = 0; j < df->col; j++) printf("%*.2f ", col_space, df->data[i][j]);
        printf("\n");
    }
}
void free_data_frame(Data_Frame* df) {
    for (int i = 0; i < df->col; i++) {
        free(df->features[i]);
        free(df->data[i]);
    }
    free(df->features);
    free(df->data);
    free(df);
}

// Numpy:

float** new_matrix(int row, int col) {
  	float** newm = (float**)malloc(row* sizeof(float*));
  	for (int i = 0; i < row; i++) newm[i] = (float*)calloc(col, sizeof(float));
  	return newm;
}
float** matrix_multiply(float** a, float** b, int row_a, int col_a, int col_b) {
  	float** res = (float**)malloc(row_a* sizeof(float*));
  	float sum;
  	for (int i = 0, j, k; i < row_a; i++) {
    		res[i] = (float*)malloc(col_b* sizeof(float));
    		for (j = 0; j < col_b; j++) {
      			sum = 0.0;
      			for (k = 0; k < col_a; k++) sum += a[i][k]* b[k][j];
      			res[i][j] = sum;
    		}
  	}
  	return res;
}
float** transpose_matrix(float** matrix, int row, int col) {
  	float** trp = (float**)malloc(col* sizeof(float*));
  	for (int i = 0, j; i < col; i++) {
    		trp[i] = (float*)malloc(row* sizeof(float));
    		for (j = 0; j < row; j++) trp[i][j] = matrix[j][i];
  	}
  	return trp;
}
void free_matrix(float** matrix, int row) {
  	for (int i = 0; i < row; i++) free(matrix[i]);
  	free(matrix);
}
float mean(float* y, int length) {
  	float total = 0.0;
  	for (int i = 0; i < length; i++) total += y[i];
  	return total / length;
}
float sum_square_error(float* y_pred, float* y_true, int length) {
  	float sum = 0.0;
  	for (int i = 0; i < length; i++) sum += (y_pred[i] - y_true[i])*(y_pred[i] - y_true[i]);
  	return sum;
}
float mean_square_error(float* y_pred, float* y_true, int length) {
	  return sum_square_error(y_pred, y_true, length) / length;
}
float sqroot(float num, float error) {
  	if (num < 0) return 0;
  	float sqr = num / 2;
  	while (sqr*sqr - num > error || num - sqr*sqr > error) sqr = (sqr + num / sqr) / 2;
  	return sqr;
}
