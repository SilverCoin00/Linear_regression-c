#include "Core.h"

typedef struct Weights {
	float* weights;    // weights and bias
	int num_weights;
} Weights;

Weights* init_weights(int num_of_features, int random_init) {
	Weights* newb = (Weights*)malloc(sizeof(Weights));
	newb->num_weights = num_of_features + 1;                               // spare 1 for bias
	newb->weights = (float*)calloc(num_of_features + 1, sizeof(float));
	srand(random_init);
	for (int i = 0; i < newb->num_weights; i++) newb->weights[i] = ((float)rand() / RAND_MAX)* 0.2 - 0.1;
	return newb;
}
float* weights_derivative(Dataset* data, Weights* w) {             // deriv(w) = X(T).(X.w - y)
	int i;
	float* deriv = (float*)malloc(w->num_weights* sizeof(float));
	float** weights = new_matrix(w->num_weights, 1);
	for (i = 0; i < w->num_weights; i++) weights[i][0] = w->weights[i];
	float** error = matrix_multiply(data->x, weights, data->samples, data->features + 1, 1);
	free_matrix(weights, w->num_weights);

	for (i = 0; i < data->samples; i++) error[i][0] -= data->y[i];
	
	float** x_T = transpose_matrix(data->x, data->samples, data->features + 1);
	float** t = matrix_multiply(x_T, error, data->features + 1, data->samples, 1);
	free_matrix(x_T, data->features + 1);
	free_matrix(error, data->samples);
	for (i = 0; i < data->features + 1; i++) {
		deriv[i] = t[i][0];
		free(t[i]);
	}
	free(t);
	return deriv;
}
void grad_descent(Dataset* data, Weights* w, float learning_rate) {
	float* gradient = weights_derivative(data, w);
	for (int i = 0; i < w->num_weights; i++) w->weights[i] -= learning_rate* gradient[i];
	free(gradient);
}
void grad_descent_momentum(Dataset* data, Weights* w, float learning_rate, float* pre_velocity, float velocity_rate) {
	float* velo = weights_derivative(data, w);
	for (int i = 0; i < w->num_weights; i++) {
		velo[i] += velocity_rate* pre_velocity[i];
		w->weights[i] -= learning_rate* velo[i];
		pre_velocity[i] = velo[i];
	}
	free(velo);
}
void print_weights(Weights* w, int decimal) {
	printf("Weights: [");
	for (int i = 0; i < w->num_weights - 1; i++) printf("%.*f, ", decimal, w->weights[i]);
	printf("%.*f]\n", decimal, w->weights[w->num_weights - 1]);
}
void free_weights(Weights* w) {
	free(w->weights);
	free(w);
}
