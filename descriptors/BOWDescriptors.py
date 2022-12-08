import cv2, numpy as np, random
from tqdm import tqdm


def SURFBOWFeatures(images_list):
    clusters = 50
    params = dict(algorithm = 1, trees = 5)
    extractor = cv2.xfeatures2d.SURF_create()
    matcher = cv2.FlannBasedMatcher(params, {})
    BOWKmeans = cv2.BOWKMeansTrainer(clusters)
    BOWExtractor = cv2.BOWImgDescriptorExtractor(extractor, matcher)
    
    training_fraction = 0.3
    index = int(len(images_list) * training_fraction)
    for i in range(index):
        img = cv2.cvtColor(images_list[random.choice(range(index))], cv2.COLOR_BGR2GRAY)
        _, descriptor = extractor.compute(img, extractor.detect(img))
        
        if descriptor is not None:
            BOWKmeans.add(descriptor)
        
    print("Clustering...")
    vocabulary = BOWKmeans.cluster()
    BOWExtractor.setVocabulary(vocabulary)
    
    print('Processing BOW features with SURF...')
    features = []
    for i in tqdm(range(len(images_list))):
        f = []
        img = cv2.cvtColor(images_list[i], cv2.COLOR_BGR2GRAY)
        descriptor = BOWExtractor.compute(img, extractor.detect(img))
        if (descriptor is None):
            f.append(np.zeros((clusters)))
        else:   
            f.append(descriptor.ravel())
    
    
        features.extend(f)
        
    return features


def SIFTBOWFeatures(images_list):
    clusters = 50
    params = dict(algorithm = 1, trees = 5)
    extractor = cv2.SIFT_create()
    matcher = cv2.FlannBasedMatcher(params, {})
    BOWKmeans = cv2.BOWKMeansTrainer(clusters)
    BOWExtractor = cv2.BOWImgDescriptorExtractor(extractor, matcher)
    
    training_fraction = 0.3
    index = int(len(images_list) * training_fraction)
    for i in range(index):
        img = cv2.cvtColor(images_list[random.choice(range(index))], cv2.COLOR_BGR2GRAY)
        _, descriptor = extractor.compute(img, extractor.detect(img))
        
        if descriptor is not None:
            BOWKmeans.add(descriptor)
        
    print("Clustering...")
    vocabulary = BOWKmeans.cluster()
    BOWExtractor.setVocabulary(vocabulary)
    
    print('Processing BOW features with SIFT...')
    features = []
    for i in tqdm(range(len(images_list))):
        f = []
        img = cv2.cvtColor(images_list[i], cv2.COLOR_BGR2GRAY)
        descriptor = BOWExtractor.compute(img, extractor.detect(img))
        if (descriptor is None):
            f.append(np.zeros((clusters)))
        else:   
            f.append(descriptor.ravel())
    
    
        features.extend(f)
        
    return features

    