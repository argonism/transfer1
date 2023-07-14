#!/usr/bin/env python
# coding: utf-8

# # Experiment on NTCIR-17 Transfer Task Eval Dataset
# 
# This notebook shows how to apply BM25 to the eval dataset of NTCIR-17 Transfer Task using [PyTerrier](https://pyterrier.readthedocs.io/en/latest/) (v0.9.2).

# ## Previous Step
# 
# - `preprocess-transfer1-eval-ipynb`
# 
# ## Requirement
# 
# - Java v11

# ## Path

# In[115]:


import os
os.environ['INDEX'] = '../indexes/ntcir17-transfer/jance'
os.environ['RUN'] = '../runs/ntcir17-transfer/jance'


# ## Datasets

# In[116]:


import sys
get_ipython().system('{sys.executable} -m pip install -q ir_datasets')


# In[117]:


sys.path.append(os.path.join(os.path.dirname(os.path.abspath('__file__')), '../datasets'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath('__file__')), '..'))


# In[118]:


import ir_datasets
import ntcir_transfer
dataset = ir_datasets.load('ntcir-transfer/1/eval')


# ## Tokenization
# 
# - In this example, we use [SudachiPy](https://github.com/WorksApplications/SudachiPy) (v0.5.4) + sudachidict_core dictionary + SplitMode.A
# - Other tokenizers can also be used

# In[119]:


import sys
get_ipython().system('{sys.executable} -m pip install -q sudachipy sudachidict_core')


# In[120]:


import re
import json
from sudachipy import tokenizer
from sudachipy import dictionary
tokenizer_obj = dictionary.Dictionary().create()
mode = tokenizer.Tokenizer.SplitMode.A


# In[121]:


def tokenize_text(text):
    atok = ' '.join([m.surface() for m in tokenizer_obj.tokenize(text, mode)])
    return atok


# In[122]:


tokenize_text('すもももももももものうち')


# ## Experiment

# ### PyTerrier

# In[123]:


# Change JAVA_HOME to fit your environment
JAVA_HOME = '/usr/lib/jvm/default'
os.environ['JAVA_HOME'] = JAVA_HOME
os.getenv('JAVA_HOME')


# In[124]:


import sys
# !{sys.executable} -m pip install -q python-terrier


# In[125]:


import pandas as pd
import pyterrier as pt
if not pt.started():
  pt.init(tqdm='notebook')


# In[ ]:





# In[126]:


dataset_pt = pt.get_dataset('irds:ntcir-transfer/1/eval')


# ### Indexing

# In[127]:


# !rm -rf $INDEX
get_ipython().system('mkdir -p $INDEX')


# In[128]:


# indexer = pt.IterDictIndexer(os.getenv('INDEX'))
# indexer.setProperty("tokeniser", "UTFTokeniser")
# indexer.setProperty("termpipelines", "")
from pathlib import Path
from importlib import reload
import models
from models.jance.jance import PyTDenseIndexer, PyTDenseRetrieval
reload(models.jance.jance)
from models.jance.jance import PyTDenseIndexer, PyTDenseRetrieval

indexer = PyTDenseIndexer(Path(os.getenv('INDEX')), verbose=False)


# In[129]:


def train_doc_generate():
    for doc in dataset.docs_iter():
        yield { 'docno': doc.doc_id, 'text': tokenize_text(doc.text) }


# In[130]:


get_ipython().run_cell_magic('time', '', 'index_path = indexer.index(train_doc_generate())\n')


# In[ ]:


get_ipython().system('ls $INDEX')


# ### Topics

# In[ ]:


def tokenize_topics():
    import re
    code = re.compile('[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％]')
    queries = dataset_pt.get_topics(tokenise_query=False)
    for idx, row in queries.iterrows():
        queries.iloc[idx, 1] = code.sub('', tokenize_text(row.query))
    return queries


# In[ ]:


tokenize_topics()


# ### Retrieval
# 
# - The performance value (e.g., nDCG) is expected to be 0.0.
# - You can use the generated run files for submission.

# In[ ]:


# Load existing index files
# indexref = pt.IndexFactory.of(os.getenv('INDEX'))


# In[ ]:


get_ipython().system('mkdir -p $RUN')


# In[ ]:


# bm25 = pt.BatchRetrieve(indexref, wmodel="BM25")
jance = PyTDenseRetrieval(index_path)


# In[ ]:


# dummy qrels
import pandas as pd
dummy_qrels = pd.DataFrame(dataset_pt.get_topics(), columns=['qid'])
dummy_qrels['docno'] = 'docno'
dummy_qrels['label'] = 0


# In[ ]:


get_ipython().run_cell_magic('time', '', 'from pyterrier.measures import *\npt.Experiment(\n    [jance],\n    tokenize_topics(),\n    dummy_qrels,\n    eval_metrics=[nDCG],\n    names = ["MyRun-BM25"],\n    save_dir = os.getenv(\'RUN\'),\n    save_mode = "overwrite"\n)\n')


# In[ ]:


get_ipython().system('gunzip -c $RUN/MyRun-BM25.res.gz | head')

