import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def load_data(filepath):
	return pd.read_csv(filepath, sep=';')


def add_noise(df, col_name, seed):
	len_rows = df.shape[0]
	np.random.seed(seed)
	partY = np.random.standard_normal(len_rows)
	df[col_name] = partY
	return df

def generate_corr(df, col_name):
	corr = df.corr()[col_name].abs()
	plt.matshow(df.corr())
	plt.show()
	return corr#.where(corr < 0.05)

def clean_data(df):
	#converts . to NaN
	df.replace({'.': np.nan}, inplace=True)
	df = df.drop(columns='Unnamed: 0')
	return df

def process_data():
	brainsize_csv_filepath = "./brainsize.csv"
	df = load_data(brainsize_csv_filepath)
	df = clean_data(df)
	df = add_noise(df, 'partY', 8)
	corr = generate_corr(df, 'partY')
	print(corr)
	df2 = add_noise(df, 'partY2', 3)
	corr2 = generate_corr(df2, 'partY2')
	print(corr2)
	return df

def main():
	print("Running Cat Boucher's Analysis!")
	results = process_data()
	#results = results.groupby('Weight').mean()
	#print(results)


if __name__ == "__main__":
	main()
