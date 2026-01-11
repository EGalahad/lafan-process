# LAFAN1 Retargeting Tools

## 1. Download Dataset and Models

```bash
# 1. Download Dataset
git clone https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset

# 2. Download Models
git clone https://huggingface.co/elijahgalahad/g1_xmls
```

## 2. Usage of Scripts


```bash
uv run visualize --csv LAFAN1_Retargeting_Dataset/g1/dance1_subject2.csv
```

Default MJCF path is `g1_xmls/g1.xml`. Optional: override the MJCF path.

```bash
G1_MJCF_PATH=/path/to/g1.xml uv run visualize --csv LAFAN1_Retargeting_Dataset/g1/dance1_subject2.csv
```

### 2.2 Export Motion to HDMI Format

```bash
uv run export --csv-folder LAFAN1_Retargeting_Dataset/g1 --out-dir output_motion
```

This produces:

```
output_motion/<motion_name>/motion.npz
output_motion/<motion_name>/meta.json
```

### 2.3 Split Motions into 1000-Frame Chunks

```bash
uv run split --input-dir output_motion --output-dir output_motion_chunks --chunk-len 1000
```

This produces:

```
output_motion_chunks/<motion_name>/0-1000/motion.npz
output_motion_chunks/<motion_name>/0-1000/meta.json
output_motion_chunks/<motion_name>/1000-2000/...
```
