import unittest
import peewee
from torch.utils.data import DataLoader

from orm import Meme, MemeType
from dataset import MemeDataset


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


if __name__ == '__main__':
    unittest.main()
