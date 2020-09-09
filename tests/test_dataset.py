import pytest
from torch.utils.data import DataLoader


def test_dataset_getter(dataset):
    for key in ['title', 'm_img', 'meme_path']:
        assert key in dataset[0]


def test_dataloader(dataset):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=True)
    for mini_batch, data in enumerate(dataloader):
        m_img = data['m_img']
        title = data['title']
        meme_path = data['meme_path']
        print("{:d}: {} in {}".format(mini_batch, title, meme_path))


if __name__ == '__main__':
    pytest.main()
