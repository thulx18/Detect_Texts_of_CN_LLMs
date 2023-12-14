from modelscope.msdatasets import MsDataset
ds = MsDataset.load('simpleai/HC3', subset_name='finance', split='train')
print(ds)