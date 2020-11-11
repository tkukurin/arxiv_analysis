import logging
import json

import spacy
import pandas as pd

import typing as ty
import _io

from sklearn.preprocessing import LabelEncoder


L = logging.getLogger(__name__)

TDictTransform = ty.Callable[[dict], dict]


def jsonl(
    file: _io.TextIOWrapper,
    transform: TDictTransform = lambda x: x,
    limit: int = None) -> ty.List[dict]:
  '''Load a `jsonl` file (each line assumed to be valid JSON).
  transform: transformation to apply for each line. Skips entry if `None`.
  '''
  count = 0
  for line, d in enumerate(map(json.loads, file)):
    if limit and count >= limit:
      break
    d = transform(d)
    if d is not None:
      count += 1
      yield d


def arxiv2df(
    arxiv_list: ty.Iterable[dict],
    drop_cols = None,
    parse_cols = None,
  ) -> ty.Tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
  '''Cleans up a collection of ArXiv docs and converts to `pd.DataFrame`.
  '''
  if not arxiv_list: raise Exception('empty list given')

  # clear up a few hundred megs of memory
  if drop_cols is None:
    drop_cols = ['versions', 'authors_parsed', 'doi', 'license', 'report-no']
  if parse_cols is None:
    parse_cols = ['abstract','authors','title','comments','submitter','journal-ref']

  df = pd.DataFrame(arxiv_list).set_index('id').drop(columns=drop_cols)
  df.update_date = pd.DatetimeIndex(df.update_date)
  df.abstract = df.abstract.apply(str.strip)

  nlp = spacy.load('en')
  for col in parse_cols:
    df[col] = df[col].apply(lambda d: nlp.make_doc(d) if d else None)

  idx_le = LabelEncoder()
  df.index = idx_le.fit_transform(df.index)

  catsall = set()
  cats_normalized = df.categories.apply(str.lower).apply(str.split)
  for c in cats_normalized:
    catsall.update(c)

  cats_le = LabelEncoder().fit(list(catsall))
  df.categories = cats_normalized.apply(cats_le.transform)

  return df, cats_le, idx_le


class Criterion(set):
  '''Selection criterion'''
  @classmethod
  def not_(cls):
    return lambda x: Not(cls(x))

class Not(Criterion):
  def __call__(self, other: ty.Iterable):
    return not super()(other)

class All(Criterion):
  '''Usage: All([set])([other set])'''
  def __call__(self, other: ty.Iterable):
    return not self.difference(other)

class Any(Criterion):
  '''Usage: Any([set])([other set])'''
  def __call__(self, other: ty.Iterable):
    return len(self.intersection(other)) > 0


class ArxivDataset:

  @staticmethod
  def load(loc='/kaggle/input/arxiv/arxiv-metadata-oai-snapshot.json', **kw):
    L.info('Loading from: %s', loc)
    with open(loc) as fp:
      arxiv_list = jsonl(fp, **kw)
      return ArxivDataset(*arxiv2df(arxiv_list))

  def __init__(self,
      df: pd.DataFrame,
      cats_le: LabelEncoder,
      idx_le: LabelEncoder):
    self.df = df
    self.cats_le = cats_le
    self.idx_le = idx_le

  @property
  def indices(self):
    return set(self.idx_le.classes_)

  @property
  def categories(self):
    return set(self.cats_le.classes_)

  def __getitem__(self, i): # return raw
    _r = self.df.iloc[i].copy()
    if isinstance(_r, pd.Series): # i is single index
      _r.categories = self.cats_le.inverse_transform(_r.categories)
      _rid = self.idx_le.inverse_transform([_r.name])
      _r = _r.append(pd.Series(_rid, index=['id'] * len(_rid)))
    else:
      _r.categories = _r.categories.apply(self.cats_le.inverse_transform)
      _r.index = self.idx_le.inverse_transform(_r.index)
    return _r

  def _wdf(self, df): # wrap df
    return ArxivDataset(df, self.cats_le, self.idx_le)

  def bycat(self, *cats: str, criterion:Criterion = Any):
    '''Get category(ies) from DF.'''
    select = criterion(self.cats_le.transform(cats))
    return self._wdf(self.df[self.df.categories.apply(select)])

# vim: shiftwidth=2 softtabstop=2 expandtab
