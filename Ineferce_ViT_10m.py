import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
from ViT_Train import make_model
import os





CKPT_PATH = "ViT_10m.pt"

IMG_PATH  = os.path.expanduser("~/Pictures/Screenshots/testmodel2.png")

IMG_SIZE  = 224

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"



CLASS_NAMES = ['bonnet, poke bonnet', 'green mamba', 'langur', 'Doberman, Doberman pinscher', 'gyromitra', 'Saluki, gazelle hound', 'vacuum, vacuum cleaner', 'window screen', 'cocktail shaker', 'garden spider, Aranea diademata', 'garter snake, grass snake', 'carbonara', 'pineapple, ananas', 'computer keyboard, keypad', 'tripod', 'komondor', 'American lobster, Northern lobster, Maine lobster, Homarus americanus', 'bannister, banister, balustrade, balusters, handrail', 'honeycomb', 'tile roof', 'papillon', 'boathouse', 'stinkhorn, carrion fungus', 'jean, blue jean, denim', 'Chihuahua', 'Chesapeake Bay retriever', 'robin, American robin, Turdus migratorius', 'tub, vat', 'Great Dane', 'rotisserie', 'bottlecap', 'throne', 'little blue heron, Egretta caerulea', 'rock crab, Cancer irroratus', 'Rottweiler', 'lorikeet', 'Gila monster, Heloderma suspectum', 'head cabbage', 'car wheel', 'coyote, prairie wolf, brush wolf, Canis latrans', 'moped', 'milk can', 'mixing bowl', 'toy terrier', 'chocolate sauce, chocolate syrup', 'rocking chair, rocker', 'wing', 'park bench', 'ambulance', 'football helmet', 'leafhopper', 'cauliflower', 'pirate, pirate ship', 'purse', 'hare', 'lampshade, lamp shade', 'fiddler crab', 'standard poodle', 'Shih-Tzu', 'pedestal, plinth, footstall', 'gibbon, Hylobates lar', 'safety pin', 'English foxhound', 'chime, bell, gong', 'American Staffordshire terrier, Staffordshire terrier, American pit bull terrier, pit bull terrier', 'bassinet', 'wild boar, boar, Sus scrofa', 'theater curtain, theatre curtain', 'dung beetle', 'hognose snake, puff adder, sand viper', 'Mexican hairless', 'mortarboard', 'Walker hound, Walker foxhound', 'red fox, Vulpes vulpes', 'modem', 'slide rule, slipstick', 'walking stick, walkingstick, stick insect', 'cinema, movie theater, movie theatre, movie house, picture palace', 'meerkat, mierkat', 'kuvasz', 'obelisk', 'harmonica, mouth organ, harp, mouth harp', 'sarong', 'mousetrap', 'hard disc, hard disk, fixed disk', 'American coot, marsh hen, mud hen, water hen, Fulica americana', 'reel', 'pickup, pickup truck', 'iron, smoothing iron', 'tabby, tabby cat', 'ski mask', 'vizsla, Hungarian pointer', 'laptop, laptop computer', 'stretcher', 'Dutch oven', 'African hunting dog, hyena dog, Cape hunting dog, Lycaon pictus', 'boxer', 'gasmask, respirator, gas helmet', 'goose', 'borzoi, Russian wolfhound']

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)

STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

tf = T.Compose([
    T.Resize(int(IMG_SIZE * 256 / 224)),
    T.CenterCrop(IMG_SIZE),
    T.ToTensor(),
])



def load_image(path, device):

    img = Image.open(path).convert("RGB")

    x = tf(img).unsqueeze(0).to(device)

    x = (x - MEAN.to(device)) / STD.to(device)

    return x


def main():

    
    model = make_model().to(DEVICE)

    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)

    model.load_state_dict(ckpt["model"])

    model.eval()


    
    x = load_image(IMG_PATH, DEVICE)

    print("input shape:", tuple(x.shape))


    
    with torch.no_grad():

        logits, aux = model(x)

        probs = F.softmax(logits, dim=-1)

        top5_p, top5_i = probs.topk(5, dim=-1)



    print("\nTop-5 predictions:")

    for rank, (p, i) in enumerate(zip(top5_p[0], top5_i[0]), 1):

        name = CLASS_NAMES[i.item()] if i.item() < len(CLASS_NAMES) else f"class_{i.item()}"

        print(f"  {rank}. {name:<20s} {p.item()*100:6.2f}%")


    
    best_i = top5_i[0, 0].item()

    best_p = top5_p[0, 0].item()

    best_name = CLASS_NAMES[best_i] if best_i < len(CLASS_NAMES) else f"class_{best_i}"

    print(f"\nPrediction: {best_name}  ({best_p*100:.2f}%)")



if __name__ == "__main__":
    
    main()