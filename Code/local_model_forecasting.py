import pandas as pd
import torch
import os
import datetime as dt
from model_CNN import get_model
import numpy as np

class DataProcessor:
    def __init__(self, main_path):
        self.main_path = main_path

    def process_file(self, filepath, target_date):
        try:
            df = pd.read_csv(filepath, dtype={'Station': str, 'Site_Name': str})
            df = df[df['Station'] != '404KE1010002']
            replacements = {'404KE1010001': 'Nairobi', '634QA1010001': 'Doha', '818EGY010001': 'Cairo'}
            df['Site_Name'] = df['Station'].map(replacements).fillna(df['Site_Name'])
            df['UTC_DATE'] = pd.to_datetime(df['UTC_DATE'])
            file_date = dt.datetime.strptime(target_date, '%Y%m%d')
            df['Forecasting_Index'] = (df['UTC_DATE'] - file_date).dt.days
            df['AirNowPM2.5'] = -999
            required_columns = ['AirNowPM2.5', 'PS', 'QV10m', 'Q500', 'T10m', 'T500', 'T850', 'WIND', 'BCSMASS',
                                'DUSMASS25', 'OCSMASS', 'SO2SMASS', 'SO4SMASS', 'SSSMASS25', 'NISMASS25', 'TOTEXTTAU',
                                'Forecasting_Index']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"Missing columns in file: {filepath}: {missing_columns}")
                return None
        except pd.errors.EmptyDataError:
            print(f"File {filepath} is empty or corrupted.")
            return None

        site_data_dict = {}
        for site in df['Site_Name'].unique():
            if site == '0':
                continue
            inputs_list = [[] for _ in range(3)]
            for index in range(3):
                index_data = df[(df['Site_Name'] == site) & (df['Forecasting_Index'] == index)]
                if index_data.empty:
                    continue
                input_tensor = torch.tensor(index_data[required_columns[1:-1]].values, dtype=torch.float32).view(-1, 8, 15)
                inputs_list[index].append(input_tensor)
            if any(inputs_list):
                inputs = torch.cat([torch.cat(items, dim=0).unsqueeze(0) for items in inputs_list if items], dim=0)
                site_data_dict[site] = {'inputs': inputs.permute(1, 0, 2, 3)}
        return df, site_data_dict

class ModelRunner:
    def __init__(self, main_path):
        self.main_path = main_path

    def load_models(self, model_path, model_names):
        models = []
        for name in model_names:
            model = get_model('batchNormLeakyDropoutCNN')
            model_path_full = os.path.join(model_path, name)
            state_dict = torch.load(model_path_full, map_location=torch.device('cpu'))
            model.load_state_dict(state_dict)
            model.eval()
            models.append(model)
        return tuple(models)

    def run_models(self, df, site_input_data, min_vals_adjusted, max_vals_adjusted, local_region):
        model_path = os.path.join(self.main_path, 'local_model', local_region)
        
        # Complete model names dictionary for all regions
        model_names = {
            'global': ['batchNormLeakyDropoutCNN_it_1_day_0_batch_128_epoch_1000.pth', 
                         'batchNormLeakyDropoutCNN_it_3_day_1_batch_128_epoch_1000.pth', 
                         'batchNormLeakyDropoutCNN_it_3_day_2_batch_128_epoch_1000.pth'],
            'africa': ['batchNormLeakyDropoutCNN_it_0_day_0_batch_64_epoch_300_Africa.pth', 
                         'batchNormLeakyDropoutCNN_it_3_day_1_batch_64_epoch_300_Africa.pth', 
                         'batchNormLeakyDropoutCNN_it_0_day_2_batch_64_epoch_300_Africa.pth'],
            'middle_east': ['batchNormLeakyDropoutCNN_it_1_day_0_batch_32_epoch_150_Middle_East.pth', 
                              'batchNormLeakyDropoutCNN_it_0_day_1_batch_32_epoch_150_Middle_East.pth', 
                              'batchNormLeakyDropoutCNN_it_1_day_2_batch_32_epoch_300_Middle_East.pth'],
            'south_asia': ['batchNormLeakyDropoutCNN_it_1_day_0_batch_64_epoch_300_South_Asia.pth', 
                             'batchNormLeakyDropoutCNN_it_1_day_1_batch_64_epoch_300_South_Asia.pth', 
                             'batchNormLeakyDropoutCNN_it_2_day_2_batch_64_epoch_300_South_Asia.pth'],
            'southeast_asia': ['batchNormLeakyDropoutCNN_it_1_day_0_batch_32_epoch_300_Southeast_Asia.pth',
                                 'batchNormLeakyDropoutCNN_it_0_day_1_batch_32_epoch_300_Southeast_Asia.pth',
                                 'batchNormLeakyDropoutCNN_it_1_day_2_batch_32_epoch_300_Southeast_Asia.pth'],
            'east_asia': ['batchNormLeakyDropoutCNN_it_2_day_0_batch_16_epoch_50_East_Asia.pth', 
                            'batchNormLeakyDropoutCNN_it_3_day_1_batch_16_epoch_100_East_Asia.pth', 
                            'batchNormLeakyDropoutCNN_it_2_day_2_batch_16_epoch_300_East_Asia.pth'],
            'central_asia': ['batchNormLeakyDropoutCNN_it_0_day_0_batch_64_epoch_150_Central_Asia.pth', 
                               'batchNormLeakyDropoutCNN_it_2_day_1_batch_64_epoch_300_Central_Asia.pth', 
                               'batchNormLeakyDropoutCNN_it_2_day_2_batch_64_epoch_300_Central_Asia.pth'],
            'europe': ['batchNormLeakyDropoutCNN_it_1_day_0_batch_16_epoch_150_Europe.pth', 
                         'batchNormLeakyDropoutCNN_it_0_day_1_batch_16_epoch_150_Europe.pth', 
                         'batchNormLeakyDropoutCNN_it_2_day_2_batch_16_epoch_300_Europe.pth'],
            'north_america': ['batchNormLeakyDropoutCNN_it_3_day_0_batch_16_epoch_300_North_America.pth', 
                                'batchNormLeakyDropoutCNN_it_1_day_1_batch_16_epoch_300_North_America.pth', 
                                'batchNormLeakyDropoutCNN_it_2_day_2_batch_16_epoch_300_North_America.pth'],
            'south_america': ['batchNormLeakyDropoutCNN_it_1_day_0_batch_16_epoch_200_South_America.pth', 
                                'batchNormLeakyDropoutCNN_it_1_day_1_batch_16_epoch_300_South_America.pth', 
                                'batchNormLeakyDropoutCNN_it_3_day_2_batch_16_epoch_300_South_America.pth']
        }

        # Ensure the local region exists in the dictionary
        if local_region not in model_names:
            raise ValueError(f"Model names for region '{local_region}' not found.")
        
        models = self.load_models(model_path, model_names[local_region])

        site_combined_tensors = {}
        for site, data in site_input_data.items():
            site_inputs = data['inputs']
            site_tensor = site_inputs.view(-1, site_inputs.shape[-1])
            normalized_site_data = (site_tensor - min_vals_adjusted) / (max_vals_adjusted - min_vals_adjusted)
            normalized_site_data = normalized_site_data.view(site_inputs.shape)

            outputs = [model(normalized_site_data) for model in models]
            combined_tensor = torch.cat([output.view(-1, 1) for output in outputs], dim=0)
            site_combined_tensors[site] = combined_tensor

        if '3HR_PM_CONC_CNN' not in df.columns:
            df['3HR_PM_CONC_CNN'] = None

        for site, tensor in site_combined_tensors.items():
            if site in df['Site_Name'].values:
                flat_tensor = tensor.detach().numpy().flatten()
                site_rows = df['Site_Name'] == site
                num_rows = site_rows.sum()
                repeated_tensor = np.tile(flat_tensor, (num_rows // len(flat_tensor) + 1))[:num_rows]
                df.loc[site_rows, '3HR_PM_CONC_CNN'] = repeated_tensor

        columns_to_remove = ['AirNowPM2.5', 'Forecasting_Index']
        df.drop(columns=columns_to_remove, axis=1, inplace=True)
        return df
class MainRunner:
    def __init__(self, main_path):
        self.main_path = main_path

    def run_main(self, target_date, region='0_global'):
        data_processor = DataProcessor(self.main_path)
        model_runner = ModelRunner(self.main_path)

        date_set = target_date.strftime("%Y%m%d")
        print('Running on ' + date_set)

        csv_path = os.path.join(self.main_path, 'output', f"{date_set}_pred.csv")
        df, site_input_data = data_processor.process_file(csv_path, date_set)

        factor_min_path = os.path.join(self.main_path, 'local_model', 'min_vals_adjusted_filtered_0508_2024.pt')
        factor_max_path = os.path.join(self.main_path, 'local_model', 'max_vals_adjusted_filtered_0508_2024.pt')
        min_vals_adjusted = torch.load(factor_min_path)
        max_vals_adjusted = torch.load(factor_max_path)

        df = model_runner.run_models(df, site_input_data, min_vals_adjusted, max_vals_adjusted, region)
        if 'Site_Name' in df.columns:
            df['Site_Name'] = df['Site_Name'].apply(lambda x: str(x).replace(" ", "_").replace("'", "_").replace('-', "_").replace(",", "_"))

        df = df.sort_values(by=['Station', 'UTC_DATE', 'UTC_TIME'])
        csv_path_local = os.path.join(self.main_path, 'output_local_model', f"{date_set}_{region}_pred.csv")
        df.to_csv(csv_path_local, index=False)

if __name__ == "__main__":
    main_path = '/raid5/jseo7/geos_data/'
    today_date = dt.date.today()
    regions = ['global', 'africa', 'middle_east', 'south_asia', 'southeast_asia', 'east_asia', 'central_asia', 'europe', 'north_america', 'south_america']
    main_runner = MainRunner(main_path)
    for region in regions:
        main_runner.run_main(today_date, region)
