#include "Core.h"

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
void train(Linear_Regression* model, char* GD_type, int iteration, float learning_rate) {
	float* y_pred = (float*)malloc(model->data->samples* sizeof(float));
	float mse;
	float* pre_velo = (float*)calloc(model->weights->num_weights, sizeof(float));
	while (iteration > 0) {
		predict(model->data, model->weights, y_pred);

		if (!strcmp(GD_type, "GD")) grad_descent(model->data, model->weights, learning_rate);
		else if (!strcmp(GD_type, "GDM")) grad_descent_momentum(model->data, model->weights, learning_rate, pre_velo, 0.9);
		else if (!strcmp(GD_type, "NAG")) nesterov_accelerated_grad(model->data, model->weights, learning_rate, pre_velo, 0.9);
		
		mse = mean_square_error(y_pred, model->data->y, model->data->samples);
		printf("Iteration left: %d, MSE = %.8f\n", iteration, mse);
		print_weights(model->weights, 8);
		iteration--;
	}
	free(pre_velo);
	free(y_pred);
}
void free_ln_model(Linear_Regression* model) {
	if (model->data) free(model->data);
	if (model->weights) free(model->weights);
	free(model);
}
