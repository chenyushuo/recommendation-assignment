# HeRec & HGTRec

Implementation of HeRec and its extension HGTRec via RecBole.

[[HeRec.py]](recbole/model/general_recommender/herec.py) [[HGTRec.py]](recbole/model/general_recommender/hgtrec.py)

## Requirements

```
pytorch                   1.7.1
cudatoolkit               10.1
torch-geometric           1.7.0
```

## Dataset

Atomic files for experiments can be downloaded from [Google Drive](https://drive.google.com/file/d/14sYcCp8PJRfZpdu7MGwrSTeOkgpfdj_x/view?usp=sharing).

```
# benchmark ratings
click.train.inter
click.test.inter
click.valid.inter

# meta-path embeddings
click.uiu
click.ui_ca_iu
click.ui_ci_iu
click.iui
click.i_ca_i
click.i_ci_i

# HIN relation files
click.bca
click.bci
```

## Quick Start

Unzip `click.zip` into `dataset/click/`.

### HeRec

```bash
python run_recbole.py --model HeRec --dataset click --config_files run_herec.yaml
```

### HGTRec

```bash
python run_recbole.py --model HGTRec --dataset click --config_files run_hgtrec.yaml
```

## Results

|Method|MAE|RMSE|
|-|-|-|
|HeRec (reported)|0.8475|1.1117|
|HeRec (reproduction)|0.9026|1.1673|
|SAGERec|0.8462|1.0918|
|HGTRec|**0.7968**|**1.0428**|

## References

* HeRec [[code]](https://github.com/librahu/HERec) [[paper]](https://arxiv.org/abs/1711.10730)
* HGT [[code]](https://github.com/acbull/pyHGT) [[paper]](https://arxiv.org/abs/2003.01332)
* RecBole [[code]](https://github.com/RUCAIBox/RecBole) [[paper]](https://arxiv.org/abs/2011.01731)
