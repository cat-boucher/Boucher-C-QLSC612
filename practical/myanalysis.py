import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def load_data(filepath):
	return pd.read_csv(filepath, sep=';')


def clean_data(df):
	#converts . to NaN
	df.replace({'.': np.nan}, inplace=True)
	df = df.drop(columns='Unnamed: 0')
	df['Height'] = df['Height'].astype(float)
	df['Weight'] = df['Weight'].astype(float)
	return df


def add_noise(df, col_name, seed):
	len_rows = df.shape[0]
	np.random.seed(seed)
	partY = np.random.standard_normal(len_rows)
	df[col_name] = partY
	return df


def pearsonr_pval(x,y):
	return pearsonr(x,y)[1]


def generate_corr(df, col_name):
	corr = df.corr(method=pearsonr_pval)[col_name]
	return corr


def process_data():
	brainsize_csv_filepath = "./brainsize.csv"
	df = load_data(brainsize_csv_filepath)
	df = clean_data(df)
	seed_max = 1000000
	for seed_i in range(seed_max):
		df = add_noise(df, 'partY', seed_i)
		corr = generate_corr(df, 'partY')
		sig_corr = corr.where(corr.abs() < 0.05).dropna()
		if sig_corr.shape[0] > 5:
			print('P-Hack Successful:')
			print(corr)
			print('Iteration had size: ', sig_corr.shape[0], ' with seed: ', seed_i)
			plt.matshow(df.corr())
			plt.show()
			break
	for seed_i in range(seed_max):
		df2 = add_noise(df, 'partY2', seed_i)
		corr2 = generate_corr(df2, 'partY2')
		sig_corr2 = corr2.where(corr2.abs() >= 0.05).dropna()
		if sig_corr2.shape[0] > 6:
			print('P-Hack Successful:')
			print(corr2)
			print('Iteration had size: ', sig_corr2.shape[0], ' with seed: ', seed_i)
			break

	if seed_i == seed_max - 1:
		print('P-Hack Failed. Couldn''t find a seed')
		sys.exit(-1)
	return df

def main():
	print("Running Cat Boucher's Analysis!")
	results = process_data()


if __name__ == "__main__":
	main()
