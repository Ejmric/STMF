# STMF
Sparse Tropical Matrix Factorization (STMF)

STMF is a novel approach for matrix completion based on tropical matrix factorization. Please refer for the model's details to Omanović, A., Kazan, H., Oblak, P. & Curk, T. Sparse data embedding and prediction by tropical matrix factorization. BMC Bioinformatics 22, 89 (2021). https://doi.org/10.1186/s12859-021-04023-9

### Real data
We used the real TCGA data in our experiments from the [paper by Rappoport N. and Shamir R.](https://academic.oup.com/nar/article/46/20/10546/5123392), and the data can be downloaded from the [link](http://acgt.cs.tau.ac.il/multi_omic_benchmark/download.html). Additional preprocessing before running our experiments is provided in our paper. PAM50 data can be found on the [link](https://github.com/CSB-IG/pa3bc/tree/master/bioclassifier\_R). BRCA subtypes are collected from [CBIO portal](https://www.cbioportal.org/).

### Use
```
import STMF as stmf
model = stmf.STMF(rank=5, criterion='iterations', max_iter=500, initialization="random_vcol")
model.fit(data)
approx = model.predict_all()
```

### Additional

The implementation of the "distance correlation" measure is from the following [link](https://gist.github.com/SherazKhan/4b2fe45c50a402dd73990c98450b2c89).
