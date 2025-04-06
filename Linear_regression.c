#include "D:\Data\code_doc\AI_model_building\Linear_regression\Core.h"

typedef struct Linear_Regression {
	Dataset* data;
	Weights* weights;
} Linear_Regression;

void predict(Dataset* data, Weights* w, float* y_pred) {
	if (!y_pred) return ;
	float** weights = new_matrix(w->num_weights, 1);
	for (int i = 0; i < w->num_weights; i++) weights[i][0] = w->weights[i];
	float** yp = matrix_multiply(data->x, weights, data->samples, data->features + 1, 1);
	free_matrix(weights, w->num_weights);
	for (int i = 0; i < data->samples; i++) {
		y_pred[i] = yp[i][0];
		free(yp[i]);
	}
	free(yp);
}
void train(Linear_Regression* model, int iteration, float learning_rate) {
	float* y_pred = (float*)malloc(model->data->samples* sizeof(float));
	float mse;
	while (iteration > 0) {
		predict(model->data, model->weights, y_pred);
		update_weights(model->data, model->weights, learning_rate);
		mse = mean_square_error(y_pred, model->data->y, model->data->samples);
		printf("Iteration left: %d, MSE = %.8f\n", iteration, mse);
		print_weights(model->weights, 8);
		iteration--;
	}
	free(y_pred);
}
void free_ln_model(Linear_Regression* model) {
	if (model->data) free(model->data);
	if (model->weights) free(model->weights);
	free(model);
}