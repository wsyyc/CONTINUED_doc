# Tutorial 2: Annotating the *In situ* metabonomics data

```python
import os
import numpy as np
from scipy.stats import gaussian_kde
from CONTINUED.annotation import *
import pandas as pd

os.chdir('/data/yuchen_data/desi_scripts/data/annotation_data/result/')

work_dir = '/data/yuchen_data/desi_scripts/data/annotation_data/combined'
output_prefix = '/data/yuchen_data/desi_scripts/data/annotation_data/result/colon_cancer_desi_'

input_lipid = '/data/yuchen_data/desi_scripts/data/annotation_data/20210930.Lipid.8_samples.uniq.txt'
input_small_mol = '/data/yuchen_data/desi_scripts/data/annotation_data/20220107.combined.small_molecule.neg.uniq.txt'

input_sample_list = '/data/yuchen_data/desi_scripts/data/annotation_data/sample.list.selected.txt'
mass_cutoff = 0.02

```

### Step1: we first parse DESI data and LC-MS data


```python
sample_mass, mass_sample, mass = Parsing_Mass_Table(input_sample_list, work_dir)
```


```python
lipid = Parsing_Lipid(input_lipid)
small_mol = Parsing_Small_Molecule(input_small_mol)
```

### Step2: We then generate a file named 'mass_dis_in_samples.txt'


```python
output_sample_mass = 'mass_dis_in_samples.txt'
Print_Mass_Diff_By_Samples(sample_mass, output_sample_mass)
```


### Step3: We next utilize kde to clustering all m/z

```python
mass_index_group = Group_Mass(mass, lipid, small_mol, mass_cutoff)
mass_clustered = Clustering_Mass_by_KDE(mass_index_group, lipid, small_mol, mass_cutoff)
```

### Step4: we can generate the file 'colon_cancer_desi_.clustered_mass.table.with.anno.txt' that recoded the annotation information for all m/z across all samples

```python
Print_Clustered_Mass_By_Sample(mass_clustered, mass_sample, lipid, small_mol, output_prefix)
```

#### Each row represents an LC-MS annotated metabolite, each column represents a sample, and each cell indicates whether an m/z value in that sample has been annotated as the corresponding metabolite. If it has, the cell value is the m/z for that sample; if not, the cell value is NaN.


```python
df = pd.read_csv('colon_cancer_desi_.clustered_mass.table.with.anno.txt', index_col=0, sep='\t')
```

```python
df.head()
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ST06_20210716</th>
      <th>ST06_20211019</th>
      <th>ST08_20211019</th>
      <th>ST103_20210718</th>
      <th>ST109_20210330</th>
      <th>ST114_20210730</th>
      <th>ST118_20211222</th>
      <th>ST121_20210806</th>
      <th>ST124_20211223</th>
      <th>ST129_20201210</th>
      <th>...</th>
      <th>ST73_20210728_mass</th>
      <th>ST73_20210729_mass</th>
      <th>ST84_20211223_mass</th>
      <th>ST87_20210331_mass</th>
      <th>ST88_20210331_mass</th>
      <th>ST91_20210406_mass</th>
      <th>ST98_20210715_mass</th>
      <th>ST98_20210804_mass</th>
      <th>anno_lipid</th>
      <th>anno_small_mol</th>
    </tr>
    <tr>
      <th>Index</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>71.0133;C3 H4 O2;H;Acrylic acid</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>74.02421;C2 H5 N O2;H;Glycine</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>78.91830999999999;H Br;H;Hydrogen bromide</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>79.95662999999999;None;H;None</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 66 columns</p>
</div>




```python

```
