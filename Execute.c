#include "Core.h"

int main() {
	char file[] = "D:\\Data\\archive1\\coffee_shop_revenue.csv";
    Data_Frame* df = read_csv(file, 1000, ",");
    Dataset* ds = trans_dframe_to_dset(df, "Daily_Revenue");
    Standard_scaler* scaler = (Standard_scaler*)malloc(sizeof(Standard_scaler));
    scaler_fit(ds, scaler, "Standard_scaler");
    scaler_transform(ds, scaler, "Standard_scaler");
    //print_dataset(ds, 5, 15, 15);
    Linear_Regression* model = (Linear_Regression*)malloc(sizeof(Linear_Regression));
    model->weights = init_weights(ds->features, 5);
    model->data = ds;
    train(model, 50, 1e-4);
    free_ln_model(model);
    free_scaler(scaler, "Standard_scaler");
    free_data_frame(df);
    //free_dataset(ds);
    return 0;
}
