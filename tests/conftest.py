import pytest
from backend.response.conversation import ConversationHandler
from backend.response.direct import DirectHandler
from backend.dataset import ImportDataset
from torchvision import transforms


# return conversation handler in CPU-only mode
@pytest.fixture(scope='module')
def conversation_handler():
    return ConversationHandler('cpu')


@pytest.fixture(scope='module')
def dataset():
    transform = transforms.Compose([
        transforms.Resize((128, 64)),
        transforms.ToTensor()
    ])
    return ImportDataset('../backend/meme', transforms=transform)


@pytest.fixture(scope='module')
def direct_handler():
    return DirectHandler()