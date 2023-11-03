# -----------------------------------------------------------------
# TODO: Leica
# -----------------------------------------------------------------
# Leica raw
# train
CUDA_VISIBLE_DEVICES=0 python /gpfs1/home/mchen/projects/HistoBistro/train_k-fold.py \
  --name=20231103_leica_raw \
  --data_config=/gpfs1/home/mchen/projects/HistoBistro/experiments/20231102_leica_raw/yaml_files/aws_data_config_MSI--MSI.yaml \
  --config=/gpfs1/home/mchen/projects/HistoBistro/experiments/20231102_leica_raw/yaml_files/aws_config_MSI--MSI.yaml

# test
python /gpfs1/home/mchen/projects/HistoBistro/test_k-fold.py \
  --name=20231103_leica_raw \
  --data_config=/gpfs1/home/mchen/projects/HistoBistro/experiments/20231102_leica_raw/yaml_files/aws_data_config_MSI--MSI.yaml \
  --config=/gpfs1/home/mchen/projects/HistoBistro/experiments/20231102_leica_raw/yaml_files/aws_config_MSI--MSI.yaml

# test: trained on philips and tested on leica
python /gpfs1/home/mchen/projects/HistoBistro/test_k-fold.py \
  --name=20231103_philips_raw \
  --data_config=/gpfs1/home/mchen/projects/HistoBistro/experiments/20231102_leica_raw/yaml_files/aws_data_config_MSI--MSI.yaml \
  --config=/gpfs1/home/mchen/projects/HistoBistro/experiments/20231102_leica_raw/yaml_files/aws_config_MSI--MSI.yaml

# test: trained on resection and tested on leica
python /gpfs1/home/mchen/projects/HistoBistro/test_k-fold.py \
  --name=20231103_resection_raw \
  --data_config=/gpfs1/home/mchen/projects/HistoBistro/experiments/20231102_leica_raw/yaml_files/aws_data_config_MSI--MSI.yaml \
  --config=/gpfs1/home/mchen/projects/HistoBistro/experiments/20231102_leica_raw/yaml_files/aws_config_MSI--MSI.yaml

# test: Eghbal
python /gpfs1/home/mchen/projects/HistoBistro/test_k-fold.py \
  --name=noLT-leica-raw-Transformer-CRC_all-MSI--MSI-lr1e-05-wd1e-05 \
  --data_config=/gpfs1/home/mchen/projects/HistoBistro/aws_HistoBistro_eamidi/CRC_all/2023-11-02--21-55-26--12epochs_noLT_leica_raw/yaml_files/aws_data_config_MSI--MSI.yaml \
  --config=/gpfs1/home/mchen/projects/HistoBistro/aws_HistoBistro_eamidi/CRC_all/2023-11-02--21-55-26--12epochs_noLT_leica_raw/yaml_files/aws_config_MSI--MSI.yaml

# -----------------------------------------------------------------
# TODO: Philips
# -----------------------------------------------------------------
# Philips raw
# train
CUDA_VISIBLE_DEVICES=1 python /gpfs1/home/mchen/projects/HistoBistro/train_k-fold.py \
  --name=20231103_philips_raw \
  --data_config=/gpfs1/home/mchen/projects/HistoBistro/experiments/20231102_philips_raw/yaml_files/aws_data_config_MSI--MSI.yaml \
  --config=/gpfs1/home/mchen/projects/HistoBistro/experiments/20231102_philips_raw/yaml_files/aws_config_MSI--MSI.yaml

# test
python /gpfs1/home/mchen/projects/HistoBistro/test_k-fold.py \
  --name=20231103_philips_raw \
  --data_config=/gpfs1/home/mchen/projects/HistoBistro/experiments/20231102_philips_raw/yaml_files/aws_data_config_MSI--MSI.yaml \
  --config=/gpfs1/home/mchen/projects/HistoBistro/experiments/20231102_philips_raw/yaml_files/aws_config_MSI--MSI.yaml

# test: trained on leica and tested on philips
python /gpfs1/home/mchen/projects/HistoBistro/test_k-fold.py \
  --name=20231103_leica_raw \
  --data_config=/gpfs1/home/mchen/projects/HistoBistro/experiments/20231102_philips_raw/yaml_files/aws_data_config_MSI--MSI.yaml \
  --config=/gpfs1/home/mchen/projects/HistoBistro/experiments/20231102_philips_raw/yaml_files/aws_config_MSI--MSI.yaml

# test: trained on resection and tested on philips
python /gpfs1/home/mchen/projects/HistoBistro/test_k-fold.py \
  --name=20231103_resection_raw \
  --data_config=/gpfs1/home/mchen/projects/HistoBistro/experiments/20231102_philips_raw/yaml_files/aws_data_config_MSI--MSI.yaml \
  --config=/gpfs1/home/mchen/projects/HistoBistro/experiments/20231102_philips_raw/yaml_files/aws_config_MSI--MSI.yaml

# test: Eghbal
python /gpfs1/home/mchen/projects/HistoBistro/test_k-fold.py \
  --name=noLT-philips-raw-Transformer-CRC_all-MSI--MSI-lr1e-05-wd1e-05 \
  --data_config=/gpfs1/home/mchen/projects/HistoBistro/aws_HistoBistro_eamidi/CRC_all/2023-11-02--22-03-29--12epochs_noLT_philips_raw/yaml_files/aws_data_config_MSI--MSI.yaml \
  --config=/gpfs1/home/mchen/projects/HistoBistro/aws_HistoBistro_eamidi/CRC_all/2023-11-02--22-03-29--12epochs_noLT_philips_raw/yaml_files/aws_config_MSI--MSI.yaml

# -----------------------------------------------------------------
# TODO: resection
# -----------------------------------------------------------------
# Resection raw
CUDA_VISIBLE_DEVICES=2 python /gpfs1/home/mchen/projects/HistoBistro/train_k-fold.py \
  --name=20231103_resection_raw \
  --data_config=/gpfs1/home/mchen/projects/HistoBistro/experiments/20231102_resection_raw/yaml_files/aws_data_config_MSI--MSI.yaml \
  --config=/gpfs1/home/mchen/projects/HistoBistro/experiments/20231102_resection_raw/yaml_files/aws_config_MSI--MSI.yaml

python /gpfs1/home/mchen/projects/HistoBistro/test_k-fold.py \
  --name=20231103_resection_raw \
  --data_config=/gpfs1/home/mchen/projects/HistoBistro/experiments/20231102_resection_raw/yaml_files/aws_data_config_MSI--MSI.yaml \
  --config=/gpfs1/home/mchen/projects/HistoBistro/experiments/20231102_resection_raw/yaml_files/aws_config_MSI--MSI.yaml
