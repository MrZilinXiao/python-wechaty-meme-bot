import unittest
import peewee
import torch
from torch.utils.data import DataLoader

from orm import Meme, MemeType
from dataset import MemeDataset, allow_img_extensions
from torch.autograd import Variable
from backend.feature_extract import InceptionExtractor


class ORMTester(unittest.TestCase):
    def test_init_table(self):
        db = peewee.SqliteDatabase('test.db')
        db.connect()
        db.create_tables([Meme, MemeType], safe=True)
        db.close()
        print("Create Tables Successfully!")

    def test_insert(self):
        self.assertRaises(peewee.IntegrityError, self.insert)  # Test for CONSTRAINS

    def insert(self):
        path = './test_path'
        title = 'test title'
        tag = 'test tag'
        feature = 'test_feature'
        Meme.create(path=path, title=title, tag=tag, feature=feature)
        test_record = Meme.get(Meme.path == path)
        return test_record.title


class DatasetTester(unittest.TestCase):
    def test_dataset_getter(self):
        dataset = MemeDataset('../backend/meme/')
        self.assertTrue('title' in dataset[0])

    def test_dataloader(self):
        dataset = MemeDataset('../backend/meme/')
        dataloader = DataLoader(dataset, batch_size=3,
                                shuffle=True, num_workers=1, drop_last=False)
        for i, batched_data in enumerate(dataloader):
            m_img = batched_data['m_img']
            title = batched_data['title']
            meme_path = batched_data['meme_path']
            print(m_img, title, meme_path)


class FeatureExtractorTester(unittest.TestCase):
    def test_inception(self):
        extractor = InceptionExtractor(allow_img_extensions, 1)
        dummy_input = Variable(torch.randn(1, 3, 299, 299))
        out: torch.Tensor = extractor.get_feature(dummy_input)
        self.assertEqual(out.shape, torch.Size([1, 2048]))


if __name__ == '__main__':
    unittest.main()
