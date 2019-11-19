FILE=$1

if [ $FILE == "celeba" ]; then

    # CelebA images
    URL=https://www.dropbox.com/s/ftcx1gf6tobtw08/celeba.zip?dl=0
    ZIP_FILE=./data/celeba.zip
    mkdir -p ./data/
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./data/
    rm $ZIP_FILE

elif [ $FILE == "brats" ]; then

    # BRATS 2013 processed synthetic images
    URL=https://www.dropbox.com/s/057db0dqp4pymoa/brats.zip?dl=0
    ZIP_FILE=./data/brats.zip
    mkdir -p ./data/
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./data/
    rm $ZIP_FILE

elif [ $FILE == 'pretrained_celeba_128' ]; then

    # Fixed-Point GAN trained on CelebA (Black_Hair, Blond_Hair, Brown_Hair, Male, Young), 128x128 resolution
    URL=https://www.dropbox.com/s/es0d8q0qk29egci/pretrained_celeba_128.zip?dl=0
    ZIP_FILE=./pretrained_models/pretrained_celeba_128.zip
    mkdir -p ./pretrained_models/
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./pretrained_models/
    rm $ZIP_FILE

elif [ $FILE == 'pretrained_brats_256' ]; then

    # Fixed-Point GAN trained on BRATS 2013 synthetic dataset, 256x256 resolution
    URL=https://www.dropbox.com/s/knfmeza0ikzo9ep/pretrained_brats_syn_256_lambda0.1.zip?dl=0
    ZIP_FILE=./pretrained_models/pretrained_brats_syn_256_lambda0.1.zip
    mkdir -p ./pretrained_models/
    wget -N $URL -O $ZIP_FILE
    unzip $ZIP_FILE -d ./pretrained_models/
    rm $ZIP_FILE

else
    echo "Available arguments are celeba, brats, pretrained_celeba_128, and pretrained_brats_256."
    exit 1
fi