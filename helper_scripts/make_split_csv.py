import pandas as pd

# for processing lawson's csv files
df = pd.read_csv("../datasets/gemm/xgemm.csv")

mask_intel7 = df['label'] == 'Intel(R) Gen9 HD Graphics NEO 750'
mask_intel1 = df['label'] == 'Intel(R) Gen9 HD Graphics NEO 1150'
mask_mali = df['label'] == 'Mali-G71 5'
mask_quadro = df['label'] == 'Quadro P5000 1733'
mask_vega = df['label'] == 'Radeon RX Vega 1630'

masks = [mask_intel7, mask_intel1, mask_mali, mask_quadro, mask_vega]

for mask, name in zip(masks, ("intel750", "intel1150", "mali", "quadro", "vega")):
    df_1 = df[mask]
    df_1.to_csv(name + ".csv")

