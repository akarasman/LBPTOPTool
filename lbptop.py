
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Karasmanoglou Apostolos, akarasman@uth.gr
University of Thessaly, 2020

"""


import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# Simple lambda that simulates a simple comparison operator s(x,y) by which
# s(x,y) = 0 if x < y and s(y,x) = 1 if x >= y
isgt = lambda x,y : 1 if x >= y else 0

# Prevents devide by zero
eps = 2*sys.float_info.min

class LBPTOPTool(object):
    """
    LBPTOPTool class objects can be used to analyze dynamic textures in videos in
    order to retrieve local binary pattern (LBP) histograms as descriptors 
    (i.e a set of features) of said dynamic pattern. 
    
    More specifically this class uses a LBP-TOP srtructure to obtain a binary
    pattern sequence that is descriptive of the local texture, these are then
    aggregated in the previously described histogram. 
    
    Includes an embedded KNN classifier subclass for predictive modeling of 
    dynamic textures.

    Attributes
    ----------
    radii_ \: tuple, length = 3, default=(1,1,1)
        The radii for the circles of each of the three orthogonal plane LBP's.

    sample_points_ \: tuple, length = 3, default=(8,8,8)
        The sample points for each of the three orthogonal plane LBP's.
    
    stride_ \: tuple, length = 3, default=(1,1,1)
        The stride (step) with witch the LBP-TOP wavelet is moved along each
        axis in a single iteration.
    
    segmentation_ \: tuple, length = 3, default=(1,1,1)
        Segment the 3x3 data matrix as many times in each direction as 
        described by the tuple. This is useful in highly specific data where
        segmentation may be able to encode locality information
        
    interpolate_ \: bool, default=False
        Flag used to indicate whether or not to use bilinear interpolation to
        to approximate gray scale value intensities of sample points when
        computing orthogonal plane LBP's
    
    verbose_ \: bool, default=False
        Flag indicates whether or not to print progress messages
    
          
    Methods
    -------
    
    Public methods\: Bellow is a list of the available methods of the class and
    a very brief description of their functionallity. For a more precise
    description of their use it is recommended that you check their docstring
        
        transform \: Transforms a 3D data volume or a set of data volumes to
        LBP histogram data
            
        fit \: Fit the embeded KNN classifier object to a given training set of
        3D data volumes
        
        predict \: Predict the class labels for a single data sample or a set 
        of data samples using the fited KNN classifier
        

    Parameters
    ----------
    
        uniform \: tuple, length = 3, default=(True,True,True)   
        Limit the amount of distinct LBPs for each orthogonal plane by 
        limiting analysis to only uniform patterns (see notes).
    
        rotation_invariant \: tuple, length = 3, default=(True,True,True) 
            Limit the amount of distinct LBPs for each orthogonal plane by 
            limiting analysis to only rotationally invariant patterns (see notes).
            
        dissimilarity \: string, defaut='chi2'
            Indicates the dissimilarity measure to be used by the KNN classifier.
            Valid values are
            
                'chi2' \: Chi-squared dissimilarity measure
                  .. math:: \sum_{i=1}^{N_{bins}} (x_{i} - y_{i})^2/(x_{i} + y_{i}) 
                 
                'log-likelihood' \: Log likelihood dissimilarity measure
                 .. math:: \sum_{i=1}^{N_{bins}} x_{i}log(y_{i})
            
            where the xi, yi variables are histogram bins
        
        denoise \: bool, default=False
            Flag indicates whether or not to denoise dataset by removing Tomek 
            Links from the dataset (see notes)
    
    Notes
    -----
    
    The assumed order of planes that corresponds to the coefficients of each
    tuple argument is xy, yt, xt
    
    Uniform patterns are frequently occuring patterns in textures that have
    at most two circular transitions in their corresponding bit sequence.
    Non uniform patterns appear less often in images and are less useful in
    distinguishing patterns in image data and are thus congregated into a
    composite category.
    
    Rotationally invariant LBPs essentially rely on a reduced set of 
    representative LBPs that cannot be transformed to any other LBP in this
    reduced set via rotatation. When using rotationally invariant LBPs, each
    LBP is mapped to a unique representative pattern within this set that can
    be rotated in order to produce said LBP.
    
    Tomek links in a dataset are pairs of data samples from different classes
    that are closer (in dissimilarity function value) to each other than 
    any other sample from their own class. We can resample our dataset by 
    removing said links (deleting both samples) in hopes to improve class
    seperabillity
    
    """

    class KNNClassifier(object) :
        """
        KNNClassifier is a subclass of the LBPTOPTool class used in classifying
        dynamic textures using pattern distribution histograms to distinguish
        between different class samples 
        
        Attributes
        ----------
        
        model_data_ \: list of tuples of length = 2
        List of tuples containing the histogram data and the corresponding 
        class labels in that order.
        
        class_labels_ \: set
        Set containing every class label
        
        denoise_ \: bool
        Flag indicates whether or not to denoise/balance data by removing Tomek
        links
        
        verbose_ \: bool
        Flag indicates whether or not to print progress messages 
        
        fitted_ \: bool
        Flag indicates whether or not classifier has been fitted
        
        """
        

        def __init__(self, dissimilarity, denoise, verbose):
            
            """
            __init__:
                
            KNN subclass initializer dunder method, for help with Parameters/
            Attributes check the KNNClassifier docstring
            
            """
            
            self.model_data_ = None
            self.class_labels_ = None
            self.class_sample_num_ = None
            self.denoise_ = denoise
            self.verbose_ = verbose
            self.fitted_ = False

            if dissimilarity == 'chi2' :
                self.dissimilarity_function = LBPTOPTool.KNNClassifier.__chi2
            elif dissimilarity == 'log-likelihood':
                self.dissimilarity_function = LBPTOPTool.KNNClassifier.__log_likelihood

        @staticmethod
        def __log_likelihood(sample_hist, model_hist):
            """
            __log_likelihood \:
                
            Log likelihood dissimilarity measure of two histograms, for the
            precise formula check LBPTOPTool docstring

            Parameters
            ----------
            sample_hist \: numpy.ndarray
                The sample LBP distribution histogram to be used in 
                classification
            model_hist \: numpy.ndarray
                A single model histogram from the classifier's training set

            Returns
            -------
            float
                Dissimilarity measure computed for the two histograms
                
            """

            dissimilarity: float = 0
            
            # Normalize the histograms such that the sum of all their bins is
            # equal to one
            sample_hist_normalized = sample_hist / np.sum(sample_hist)
            model_hist_normalized = model_hist / np.sum(model_hist)
            for sample_bin, model_bin in zip(sample_hist_normalized,
                                             model_hist_normalized):
                dissimilarity -= sample_bin * np.log(model_bin + eps)

            return dissimilarity

        @staticmethod
        def __chi2(sample_hist, model_hist):
            """  
            __chi2 \:
                
            chi-square dissimilarity measure of two histograms, for the precise
            formula check LBPTOPTool docstring

            Parameters
            ----------
            sample_hist : numpy.ndarray
                The sample LBP distribution histogram to be used in 
                classification
            model_hist : numpy.ndarray
                A single model histogram from the classifier's training set

            Returns
            -------
            float
                Dissimilarity measure computed for the two histograms
            
            """


            dissimilarity: float = 0
            
            # Normalize the histograms such that the sum of all their bins is
            # equal to one
            sample_hist_normalized = sample_hist / np.sum(sample_hist)
            model_hist_normalized = model_hist / np.sum(model_hist)
            for sample_bin, model_bin in zip(sample_hist_normalized,
                                             model_hist_normalized):
                dissimilarity += \
                (sample_bin - model_bin)**2 / (sample_bin + model_bin + eps)

            return dissimilarity

        @staticmethod
        def __hist_data_dissimilarity(sample_hists, model_hists,
                                      dissimilarity_function):
            """
            __hist_data_dissimilarity \:
                
            Measures dissimilarity between two data volumes samples based on 
            their LBP distributions at different segments

            Parameters
            ----------
            sample_hists : list of numpy.ndarray
                A list of sample histograms that describe the LBP distribution
                in a certain sample data volume segment
            model_hists : list of numpy.ndarray
                A list of model histograms that describe the LBP distribution
                in a certain training set data volume segment
            dissimilarity_function : function
                Function to be used to measure dissimilarity between histograms
                (can either be chi2 or log_Likelihood)

            Returns
            -------
            float
                Total sum of dissimilarities in LBP histograms of volume data
                segments

            """
            dissimilarities = list()
            for sample_hist, model_hist in zip(sample_hists, model_hists):
                dissimilarities.append(dissimilarity_function(sample_hist,
                                                              model_hist))

            # Maybe change this to weighted sum ? A perceptron-like component
            # could be used to set the weights for the dissimilarities in
            # the LBP histograms of different segments in the data volume as
            # to improve classification accuracy by distinguishing the most
            # relevant segments
            return np.sum(dissimilarities) 

        def __get_closest_class(closest_neighboors, class_labels):
            """
            __get_closest_class \:
            
            Used to get the most likely class among the K-nearest neighboors
            of the sample to be classified.

            Parameters
            ----------
            closest_neighboors : list of float
                A sorted list of the K-nearest neighboor's dissimilarities with
                the sample to be classified
            class_labels : list
                A list of class labels of the closest neighboors

            Returns
            -------
            Label of the closest class

            """

            # "votes" for each class are aggregated in a dictionary where the
            # class labels are keys to bins containing "votes"
            votes = dict(zip(class_labels, np.zeros(len(class_labels))))
            for dissimilarity, label in closest_neighboors:
                
                # A "vote" is a float equal to the inverse value of the
                # neighboor's dissimilarity to the sample
                votes[label] += 1/(dissimilarity + eps)

            return max(votes, key = lambda k : votes[k])

        def __remove_tomek_links(self):
            """
            __remove_tomek_links \: 
                
            In order to increase class seperability,  this method "de-noises"
            the training set by removing all samples that show up in "Tomek 
            links".
            
            For more on Tomek links, see LBPTOPTool class notes

            Returns
            -------
            None.

            """
            
            # Save minimum distance indexes and minimum distance labels, these
            # will later be used to determine whether or not a pair of samples
            # is contained in a Tomek link and to be removed
            min_indexes = list(range(len(self.model_data_)))
            min_labels = [label for _, label in self.model_data_]
            
            # Phase 1 of removal procedure: Iterate across all model data while 
            # checking for Tomek links and removing them once encountered
            i = 0
            while i < len(self.model_data_) :
                

                # Pick the next model data sample (model1) to examine for 
                # possibly existing in Tomek link
                model1_hists, model1_label = self.model_data_[i]
                j = 0
                min_dissimilarity = sys.float_info.max
                
                # Find minimum dissimilarity class
                while j < len(self.model_data_):
                    
                    if j == i :
                        j += 1
                        continue
                    
                    model2_hists, model2_label = self.model_data_[j]
                    dissimilarity = \
                    LBPTOPTool.KNNClassifier.__hist_data_dissimilarity(model1_hists,
                                                                       model2_hists,
                                                                       self.dissimilarity_function)
                    
                    # Update new minimum dissimilarity if the current sample's
                    # dissimilarity is less then the previously computed one
                    if dissimilarity < min_dissimilarity :
                        min_dissimilarity = dissimilarity
                        
                        if model1_label != model2_label :
                            min_indexes[i] = j
                            min_labels[i] = model2_label
                    
                    j += 1
                
                i += 1
            
            # Phase 2 of the procedure: Re-iterate through the model data 
            # samples and remove any pair of samples that exist in Tomek Link
            # by not including them in the "clean" model data set
            clean_model_data = list()
            for i in range(len(self.model_data_)):
                if min_indexes[i] != i and \
                   min_indexes[min_indexes[i]] == i :
                    
                    j = min_indexes[i]
                    min_label = min_labels[i]
                    sample_label = self.model_data_[i][1]
                    
                    # Break Tomek links by removing samples from the majority 
                    # class
                    if self.class_sample_num_[min_label] < \
                       self.class_sample_num_[sample_label] :
                        self.class_sample_num_[sample_label] -= 1
                        clean_model_data.append(self.model_data_[j])
                    elif self.class_sample_num_[min_label] > \
                         self.class_sample_num_[sample_label] :
                        self.class_sample_num_[min_label] -= 1
                        clean_model_data.append(self.model_data_[i])
                    else :
                        self.class_sample_num_[sample_label] -= 1
                        self.class_sample_num_[min_label] -= 1
                else :
                    clean_model_data.append(self.model_data_[i])
            
            # Swap the model dataset for the clean one
            self.model_data_ = clean_model_data
            
            return self

 
        def __get_closest_neighboors(dissimilarities, neighboors):
            """
            __get_closest_neighboors \:
                
             Finds the closest neighboors of a sample given the dissimilarities
             with the other model samples.
                
            Parameters
            ----------
            dissimilarities \: list of tuples of size 2
                Pairs of dissimilarities and class labels for specific model
                data samples 
            neighboors \: int
                Number of closest neighboors to get.

            Returns
            -------
            closest_neighboors \: list
                List containing the closest neighboors of the class
            """
            
            closest_neighboors = list()
            
            for i in range(neighboors):    
                # Get next minimum distance neighboor
                min_index = np.argmin(dissimilarities)
                closest_neighboors.append(dissimilarities[min_index])
                del dissimilarities[i]
                
            return closest_neighboors
        
        def predict(self, sample, neighboors=1):
            """
            predict \:
                
            Public* predictor method. Used to predict the class labels for a
            sample or set of samples
            
            * By "public", we mean that it is visible to the LBPTOPTool class
            however the KNNClassifier is not ment to be used seperately from
            this class and therefore this method is not public in the 
            "traditional" sense
            
            Parameters
            ----------
            sample \: numpy.ndarray
                A dynamic texture's lbp histogram data
            neighboors \: int, optional, default = 1
                Number of neighboors to use in making prediction

            Returns
            -------
            predicted_class \: 
                The predicted class label
            """
            
            if not self.fitted_ :
                raise AssertionError('')
            
            # Calculate class dissimilarities for each data model sample 
            class_dissimilarities = list()
            for model, label in self.model_data_:
                diss = LBPTOPTool.KNNClassifier.__hist_data_dissimilarity(sample,
                                                                        model,
                                                                        self.dissimilarity_function)
                class_dissimilarities.append((diss, label))

            # Get the closest neighboors and use them to extract the likeliest
            # class for sample
            closest_neighboors = sorted(class_dissimilarities)[:neighboors]
            predicted_class = LBPTOPTool.KNNClassifier.__get_closest_class(closest_neighboors,
                                                                         self.class_labels_)
            
            
            return predicted_class

        def fit(self, train_samples, class_labels):
            """
            fit \:
                
            Public method* that fits the KNNClassifier object on the 
            train_samples and class_labels train data

            Parameters
            ----------
            train_samples : numpy.ndarray
                The data points of the training set
            class_labels :
                Class labels for the training set samples
            
            Raises
            ------
            AssertionError: 
                Whenever train_samples, or any single sample is not numpy.ndarray

            Returns
            -------
            self
                Fitted KNNClassifier object

            """

            self.class_labels_ = set(class_labels)
            
            # Create model dataset
            self.model_data_ = list()
            for sample, label in zip(train_samples, class_labels):
                if not isinstance(sample, np.ndarray):
                    raise AssertionError('sample' + str(sample) + 'is not ndarray')
                    
                self.model_data_.append((sample, label))
            
            self.class_sample_num_ = dict(zip(self.class_labels_,
                                             np.zeros(len(self.class_labels_))))
            
            for _, label in self.model_data_ :
                self.class_sample_num_[label] += 1
            
            if self.denoise_ :

                if self.verbose_ :
                    print('Denoising data by removing Tomek Links...')
                    num_samples_before = len(self.model_data_)

                # Denoise and balance data by removing tomek links
                self = self.__remove_tomek_links()

                if self.verbose_ :
                    num_samples_after = len(self.model_data_)
                    print('Removed',
                          str(num_samples_before - num_samples_after), \
                          'samples from the dataset')

            self.fitted_ = True
            
            return self



    def __init__(self, radii=(1,1,1), sample_points=(8,8,8), stride=(1,1,1),
                 segmentation=(1,1,1), uniform=(True, True, True),
                 rotation_invariant=(True, True, True), dissimilarity='chi2',
                 denoise=False, interpolate=True, verbose=False):
        """
        LBPTOPTool class initializer dunder method, for help with class 
        attributes see LBPTOPTool docstring

        """
        
        if any([r <= 0 for r in radii]):
            raise AssertionError('LBP sample circle radius cannot be <= 0' +
                                 str(radii))
        else:
            self.radii_ = radii
        
        if any([s <= 0 for s in sample_points]):
            raise AssertionError('Sample point number cannot be <= 0: ' + \
                                 str(sample_points))       
        else:
            self.sample_points_ = sample_points  
            
        if any([s > 16 for s in sample_points]):
            raise AssertionError('Sample point number cannot be > 16: ' + \
                                 str(sample_points))       
        else:
            self.sample_points_ = sample_points  
            
        
        if any([s <= 0 for s in stride]):
            raise AssertionError('Stride cannot be <= 0: ' + str(stride))
        else:
            self.stride_ = stride
            
        if any([s <= 0 for s in segmentation]):
            raise AssertionError('Segmentation cannot be <= 0: ' + \
                                 str(stride))
        else:    
            self.segmentation_ = segmentation
        
        self.interpolate_ = interpolate
        self.verbose_ = verbose
        
        if dissimilarity != 'log-likelihood' and \
           dissimilarity != 'chi2' :
               raise AssertionError('Supported dissimilarity functions are' + \
                                    'chi2, log-likelihood')
        
        self.classifier_ = LBPTOPTool.KNNClassifier(dissimilarity,
                                                   denoise,
                                                   verbose)

        self.lookup_xy = LBPTOPTool.__create_lbp_tag_lookup(sample_points[0],
                                                          uniform[0],
                                                          rotation_invariant[0])
        self.lookup_yt = LBPTOPTool.__create_lbp_tag_lookup(sample_points[1],
                                                          uniform[1],
                                                          rotation_invariant[1])
        self.lookup_xt = LBPTOPTool.__create_lbp_tag_lookup(sample_points[2],
                                                          uniform[2],
                                                          rotation_invariant[2])



    ## CLASS STATIC METHODS ##

    # Note: Most static methods are also private as to obscure complex backend
    # functionallity of the LBP-TOP tool from the user, in any occasion, no other
    # method other than those documented in the class docstring should be used
    # by the user

    @staticmethod
    def __rotate_binary_pattern_right(binary_pattern, pattern_len):
        """
        __rotate_binary_pattern_right \:
            
        "Rotates" a binary pattern to the right, meaning that all binary digits
        get shifted one position to the right with the rightmost (least
        significant) digit replacing the leftmost (most significant) digit.
        
        Example: 0b01101 gets rotated once to 0b10110 and

        Parameters
        ----------
        binary_pattern : int
            Integer representation of the binary pattern
        pattern_len : int
            Binary pattern length (in terms of binary digits)

        Returns
        -------
        int
            An integer representation of the rotated pattern

        """
        

        if binary_pattern & 0b1 :
            return (binary_pattern >> 1) + (1 << (pattern_len - 1))
        else:
            return (binary_pattern >> 1)

    @staticmethod
    def __create_lbp_tag_lookup(pattern_len, uniform, rotation_invariant):
        """
        __create_lbp_tag_lookup:

        Creates an lbp tag lookup that maps any lbp pattern to an lbp pattern
        tag such that patterns are uniform and rotationally invariant.
        
        For more on uniform and rotationally invariant patterns, see class 
        docstring notes.
        
        This method has an exponential running time, and therefore we limit
        the maximum number of sample points to 16 (a complete overkill in 
        most situations, i assume) such that the initialization time is 
        reasonable.
        
        Parameters
        ----------
        pattern_len : int
            Length of the binary pattern in terms of digits
        uniform : bool
            Flag indicates whether or not lookup is to contain only uniform
            patterns
        rotation_invariant : bool
            Flag indicates whether or not lookup is to contain only
            rotationally invariant patterns

        Returns
        -------
        lookup : dict or None
            A lookup (dictionary) that maps all patterns to rotationally
            invariant pattern, and/or raises KeyError whenever pattern is not
            uniform (based on what flags are present when function is called)
        """        

        lookup = None

        if rotation_invariant:
            
            # Create a lookup table that ensures rotational invariance of the
            # patterns
            lookup = dict()
            new_index = 0
            
            # Examine each possible pattern 
            for pattern in range(0, 1 << pattern_len):

                binary_pattern = pattern
                rotation_invariant_pattern = None
                
                # Select a preexisting rotationally invarient pattern from the
                # lookup
                for rotation_invariant_pattern in lookup:
                    
                    # Rotate the current pattern until it matches the
                    # rotationally invariant one
                    for rotation_num in range(pattern_len):
                        if rotation_invariant_pattern == binary_pattern :
                            break
                        else:
                            binary_pattern = LBPTOPTool.__rotate_binary_pattern_right(binary_pattern,
                                                                                pattern_len)
                   
                    # True if binary pattern matches the rotationally invariant
                    # one after a number of rotations
                    if rotation_invariant_pattern == binary_pattern :
                        break
                
                # True if no rotationally invariant pattern matches the given
                # pattern after the maximum number of rotations
                if rotation_invariant_pattern != binary_pattern :
                    lookup.update({pattern : new_index})
                    new_index += 1

            if uniform:

                # Make rotationally invariant lookup contain only uniform
                # patterns
                keys = list(lookup.keys())
                for key in keys:
                    
                    # Delete any pattern from the lookup that is not uniform
                    if LBPTOPTool.__count_circular_bit_transitions(key, pattern_len) > 2:
                        del lookup[key]

        elif uniform:

            lookup = dict()
            new_index = 0
            for pattern in range(0, 1 << pattern_len):
                
                # Reject any pattern that is not uniform from the lookup
                if LBPTOPTool.__count_circular_bit_transitions(pattern,pattern_len) <= 2:
                    lookup.update({pattern : new_index})
                    new_index += 1
        return lookup

    @staticmethod
    def __create_plane_lbp_histogram(lookup, sample_points):
        """
         __create_plane_lbp_histogram:

        Creates a histogram structure (dictionary) that describes the texture
        distribution which is nothing more then a dictionary, whose
        keys are lbp tags that index the corresponding bins that contain counts
        of each pattern
        
        Parameters
        ----------
        lookup \: dict
            Lookup that maps patterns to a rotationally invariant counterpart
            and/or rejects non uniform patterns.
        sample_points \: tuple, length = 3
            

        Returns
        -------
        hist \: numpy.ndarray
            A histogram containg all pattern occurances

        """

        # The "keyset" for both the uniform and non-uniform cases is essentialy
        # the full range of integers from zero to the length of the lookup, i.e
        # each one of the patterns contained in the lookup is assigned a unique
        # identifier integer as a key
        if lookup :
            keyset = set(range(len(lookup)))
            hist = dict(zip(keyset, np.zeros(len(keyset)))) # Bins set to zero
        else :
            keyset = range(0, 1 << sample_points)
            hist = dict(zip(keyset, np.zeros(1 << sample_points))) # Bins zet to zero

        return hist


    @staticmethod
    def __count_circular_bit_transitions(pattern, pattern_len):
        """
        __count_circular_bit_transitions:

        Counts the number of transitions in that occur in the bit string 
        "pattern" of length pattern_len. This is used in determining whether 
        the lbp is uniform or non-uniform when assigning lbp tags.
        
        The term "circular" refers to the fact that we consider the bit 
        sequence as repeating from the start once fully traversed. In other 
        words, when reaching the final bit in the sequence, the following bit
        is the once again the first one in the bit string.

        As an example consider the sequence 0b1110, this sequence has exactly 2
        transitions (0->1, followed by a sequence of 1's and then 1->0 at
        final bit). Another example: the sequence 0b1010 has 3 transitions.

        Parameters
        ----------
        pattern : int
            Integer representation of the pattern
        pattern_len : int
            Length of the pattern, in terms of binary digits

        Returns
        -------
        count : int
            The number of circular bit transitions
        """

        count = 0

        # At each instance the right most bit in the sequence is examined, the
        # sequence is traversed right to left noting each transition. The
        # prev_digit and next_digit variables contain either a zero or a one,
        # as such binary operations can be applied for efficient comparison and
        # manipulation
        prev_digit = pattern & 0b1
        for i in range(0, pattern_len):

            # We shift the sequence by a modularly increasing amount in order
            # to obtain the next righmost bit, the modulo operation is used to
            # repeat the first binary digit once the end of the sequence has
            # been reached
            next_digit = (pattern >> ((i + 1) % pattern_len)) & 0b1

            # Increment counter only if the next digit differs from the previous,
            # i.e whenever encountering a transition. Use xor for comparison.
            count += next_digit ^ prev_digit
            prev_digit = next_digit

        return count

    @staticmethod
    def __segment_texture(texture, segmentation):
        """
        __segment_texture \:

        Dynamic textures may be "segmented" in order to perform more temporally
        and locally specific analysis of the sub-volume's texture distribution.
        
        Parameters
        ----------
        texture \: numpy.ndarray
            A 3-dimensional numeric description of the dynamic texture (video
            data)
        segmentation \: tuple, length = 3
            A tuple (nx, ny, nz) that segments the dynamic texture data into
            nx, ny, and nz segments in the x,y and t directions accordingly

        Returns
        -------
        segments \: list of numpy.ndarray
            List of segments created for the data volume
        """

        # Dimensions of the sub-volume segments are calculated
        seg_len_x = int(texture.shape[0] / segmentation[0])
        seg_len_y = int(texture.shape[1] / segmentation[1])
        seg_len_t = int(texture.shape[2] / segmentation[2])

        # Construct an iterable sequence of segments
        segments = []
        for x in range(segmentation[0]):
            for y in range(segmentation[1]):
                for t in range(segmentation[2]):

                    segment = texture[x*seg_len_x:(x+1)*seg_len_x,
                                      y*seg_len_y:(y+1)*seg_len_y,
                                      t*seg_len_t:(t+1)*seg_len_t]
                    segments.append(segment)

        return segments

    @staticmethod
    def __bilinearly_interpolate_raster(plane_raster, point):
        """
        __bilinearly_interpolate_raster:

        In order to obtain a continuous gray scale gradient for the dynamic
        surface volume all real-number coefficients within the bounds of the
        volume must be  mapped to some gray scale intensity value. For this we
        use bilinear interpolation on the orthogonal plane on which the binary
        pattern lies for  each one of its sample points.

        Bilinear interpolation: Consider some arbitrary function f we wish to
        approximate. Given the values of said f at points (x0, y0), (x0, y1),
        (x1, y0), (x1, y1) which form a rectangle of area = 1 on the plane raster.
        In order to approximate the function at any point (x,y) within this area
        we obtain u,v, the fractional parts of x and y respectively and use the
        following formula to derive the gray scale value

        f(x,y) = f00*(1-u)*(1-v) + f01*(1-u)*v + f10*u*(1-v) + f11*u*v

        Where f00,f01,f10,f11 are the values at the adjacent raster points
        
        Parameters
        ----------
        plane_raster : numpy.ndarray
            A 2-dimensional array, that contains a "slice", at any of the x,y,t
            directions, of the dynamic texture data volume
        point : tuple, length = 2
            A point in the plane raster at which we want to compute the value
            of the gray scale raster

        Returns
        -------
        float
            The approximate gray scale value at "point" 

        """

        ## Get the integers for the indices that correspond to the adjacent
        ## pixel coefficients on the plane raster
        x0 = int(np.floor(point[0]))
        x1 = int(np.ceil(point[0]))
        y0 = int(np.floor(point[1]))
        y1 = int(np.ceil(point[1]))

        # Get fractional part of point coefficients
        u = point[0] - x0
        v = point[1] - y0

        # Get gray scale values of adjacent pixels
        f00 = plane_raster[(x0, y0)]
        f01 = plane_raster[(x1, y0)]
        f10 = plane_raster[(x0, y1)]
        f11 = plane_raster[(x1, y1)]

        # Τhis is a refactored version of the previously introduced formula
        # optimized to reduce the number of float multiplications
        return f00 + f01*u + f10*v + (f00 - f01 - f10 + f11)*u*v

    @staticmethod
    def __get_plane_lbp_pattern(plane, center, sample_points, radius, thresh, 
                                interpolate):
        """
        __get_plane_lbp_pattern \:

        Get lbp pattern for some orthogonal plane of the LBP-TOP wavelet,
        centered at some given point. This is done by thresholding all the points
        that lie on the periphery of the circle around some center point with
        its gray scale value. By this process we derive a binary tag that is
        used as a descriptor of the plane's texture.
        
        Parameters
        ----------
        plane \: numpy.ndarray
            A 2-dimensional "slice", at any of the x,y,t directions of the 
            dynamic texture data 
        center \: tuple, length=2
            
        sample_points \: int
            Number of sample points
        radius : float
            The radius of a sample circle 
        thresh \: float
            The threshold value with which the neighbooring sample points are 
            compared to.
        
        interpolate \: bool
            Flag indicates whether or not to use bilinear interpolation to
            approximate sample point grayscale intensities 

        Returns
        -------
        pattern : int
            Integer representation of the binary pattern
            
        """

        """
        OPTIMIZE THIS! : It is meaningful to reduce floating point operations
        here. Get rid of costly sine/cosine computation, since the points form a
        regular polygon, calculating the next point in the sequence is just a
        matter of computing the x and y differentials between the current and next
        point and simply adding these differentials to the x,y coefficients of the
        current point to get the next point. Unfortunately im too rusty in my
        geometry to easilly figure this one out.
        """

        pattern = 0
        for p in range(0, sample_points) :

            # Point coordinates computed as polar coefficients
            point = (radius*np.cos(2*np.pi*p/sample_points) + center[0],
                     radius*np.sin(2*np.pi*p/sample_points) + center[1]) # GET RID OF THIS !
            
            if interpolate :
                gray_scale_intensity = LBPTOPTool.__bilinearly_interpolate_raster(plane,
                                                                                point)
            else :
                gray_scale_intensity = plane[round(point[0]), round(point[1])]
            
            pattern += isgt(gray_scale_intensity, thresh) << p

        return pattern

    @staticmethod
    def __get_lbp_pattern(radii, sample_points, texture, center, interpolate):
        """
        __get_lbp_pattern \:

        Used to derive the lbp tag for the center point of some dynamic surface.
        In TOP-LBP each orthogonal plain has its own pattern computed
        independently of the other two and the resulting patterns are
        concatenated into a binary code used as a dynamic texture descriptor
        
        Parameters
        ----------
        radii \: tuple of float, length = 3
            Radii of the sample circles at each one of the xy, yt, xt planes
        
        sample_points \: 
            Number of sample points at each one of the xy, yt, xt planes
        
        texture \: numpy.ndarray
            A 3-dimensional numerical description for a dynamic texture
        
        center \: tuple of float, length = 3
            Center point of the LBP-TOP wavelet
        
        interpolate \: bool
            Flag indicates whether or not to use bilinear interpolation to
            approximate sample point grayscale intensities 

        Returns
        -------
        pattern_xy, pattern_yt, pattern_xt \: int
            Integer descriptions of the LBP patterns at each one of the xy, yt
            and xt planes

        """

        # Get gray-scale value of texture at center point
        gray_scale_threshold = texture[center[0], center[1], center[2]]

        
        pattern_xy = LBPTOPTool.__get_plane_lbp_pattern(texture[:,:,center[2]],
                                                      (center[0], center[1]),
                                                      sample_points[0],
                                                      radii[0],
                                                      gray_scale_threshold,
                                                      interpolate)

        pattern_yt = LBPTOPTool.__get_plane_lbp_pattern(texture[center[0],:,:],
                                                      (center[1], center[2]),
                                                      sample_points[1],
                                                      radii[0],
                                                      gray_scale_threshold,
                                                      interpolate)

        pattern_xt = LBPTOPTool.__get_plane_lbp_pattern(texture[:,center[1],:],
                                                      (center[0], center[2]),
                                                      sample_points[2],
                                                      radii[2],
                                                      gray_scale_threshold,
                                                      interpolate)

        return pattern_xy, pattern_yt, pattern_xt

    def __increment_histogram_bin_by_one(hist, lookup, pattern, pattern_len) :
        """
        __increment_histogram_bin_by_one \:

        Used every time an occurance of a certain pattern is counted and
        updates the corresponding plane histogram at the bin indexed by the
        tag assigned to this pattern in the lookup.

        """
        # If the uniform flag is set, all non-uniform patterns, i.e those with
        # more than 2 circular bit transitions are disregarded
        if lookup :
            try :
                hist[lookup[pattern]] += 1
            except KeyError :
                return hist

        else :

            # When non uniform tags are accounted for, histogram bins may be
            # accessed immediatilly without need of lookup
            hist[pattern] += 1

        return hist

    def __generate_plane_histograms(self, texture, interpolate):
        """
        __generate_plane_histograms \:

        Generate histograms for the XY, YT, and XY planes of the TOP LBPTOPTool. 
        This is done by essentialy convolving the LBP-TOP wavelet with the 
        tdynamic exture volume. Each plane's texture distributions are recorded
        in a histogram structure.
        
        Parameters
        ----------
        texture \: numpy.ndarray
            A single three-dimensional matrix or any iterable containing a set
            of three-dimensional matrices (volumes) containing the gray scale
            intensities of pixels in a video of some dynamic texture.\
        
        interpolate \: bool
            Flag indicates whether or not to use bilinear interpolation to
            approximate sample point grayscale intensities 
            
        Returns
        -------
        hist_xy, hist_yt, hist_xt \: dict
            Histograms containing bin-pattern occurance mappings for each one
            of the planes xy, yt, xt
        """
        
        """
        This method is computationally costly...
        Maybe use CUDA support ?
        """

        # Make plane lbp histograms

        hist_xy = LBPTOPTool.__create_plane_lbp_histogram(self.lookup_xy,
                                                        self.sample_points_[0])

        hist_yt = LBPTOPTool.__create_plane_lbp_histogram(self.lookup_yt,
                                                        self.sample_points_[1])

        hist_xt = LBPTOPTool.__create_plane_lbp_histogram(self.lookup_xt,
                                                        self.sample_points_[2])


        # "Slide" LBP-TOP wavelet across texture volume
        for x in range(self.radii_[0],
                       texture.shape[0] - self.radii_[0] - 1,
                       self.stride_[0]):
            for y in range(self.radii_[1],
                           texture.shape[1] - self.radii_[1] - 1,
                           self.stride_[1]):
                for t in range(self.radii_[2],
                               texture.shape[2] - self.radii_[2] - 1,
                               self.stride_[2]):

                    # Get patterns for each plane centered at an arbitrary
                    # point (x, y, t)
                    patterns = LBPTOPTool.__get_lbp_pattern(self.radii_,
                                                            self.sample_points_,
                                                            texture,
                                                            (x,y,t),
                                                            interpolate)


                    hist_xy = LBPTOPTool.__increment_histogram_bin_by_one(hist_xy,
                                                                         self.lookup_xy,
                                                                         patterns[0],
                                                                         self.sample_points_[0])

                    hist_yt = LBPTOPTool.__increment_histogram_bin_by_one(hist_yt,
                                                                         self.lookup_yt,
                                                                         patterns[1],
                                                                         self.sample_points_[1])

                    hist_xt = LBPTOPTool.__increment_histogram_bin_by_one(hist_xt,
                                                                         self.lookup_xt,
                                                                         patterns[2],
                                                                         self.sample_points_[2])

        return hist_xy, hist_yt, hist_xt

    def __transform(self, texture) :
        """
        __transform \:

        This method is used to transform a single texture to histogram data,
        the texture is segmentented into sub-volumes of dynamic textures and
        subsequently the __generate_plane_histograms method is used to derive
        the lbp histograms of the XY, YT, and XT planes for these sub-volumes.
        
        Parameters
        ----------
        texture : numpy.ndarray
            A single three-dimensional matrix or any iterable containing a set
            of three-dimensional matrices (volumes) containing the gray scale
            intensities of pixels in a video of some dynamic texture.

        Raises
        ------
        AssertionError
            Whenever texture data is not an instance of np.ndarray

        Returns
        -------
        hists_nd_array : np.ndarray
            Numpy array with transformed histogram data 


        """
        
        if not isinstance(texture, np.ndarray):
            raise AssertionError('Texture data needs to be of type np.ndarray')

        hists = []
        for segment in LBPTOPTool.__segment_texture(texture, self.segmentation_):

            hist_xy, hist_yt, hist_xt = self.__generate_plane_histograms(segment,
                                                                         self.interpolate_)

            xy_vals = np.array(list(hist_xy.values()))
            yt_vals = np.array(list(hist_yt.values()))
            xt_vals = np.array(list(hist_xt.values()))

            hists.append(np.append(xy_vals, [yt_vals, xt_vals]))
        
        hists_nd_array = np.array(hists)
        return hists_nd_array

    ## PUBLIC METHODS ##
    ## The following methods comprise the interface of the LBP-TOP object

    def transform(self, textures):
        """
        transform \:

        Used to transform any set of dynamic texture video data samples to a
        set of histograms describing the texture distributions of

        This method is essentially a wrapper for the __transform static method
        of the LBP-TOP class. It can be used to transform any set of dynamic
        texture  matrices to histogram data.

        Parameters
        ----------
        textures \:
            A single three-dimensional matrix or any iterable containing a set
            of three-dimensional matrices (volumes) containing the gray scale
            intensities of pixels in a video of some dynamic texture.

        Returns
        -------
        numpy.ndarray
            An array of histograms that describes the distribution of LBP-TOP
            patterns in each of the segments of the texture volume.
        """

        if self.verbose_ :
            print('Transforming video data to histograms...')
        
        if len(textures.shape) < 4:
            hist = self.__transform(textures)
            return np.array(hist)
        else :
            hists = []

            for texture in textures :
                hists.append(self.__transform(texture))

            return np.array(hists)

    def plot_lbp_histogram(self, texture):
        """
        plot_lbp_histogram \:
            
        Plots the LBP histogram containing the pattern distribution for a 
        given dynamic texture. For demo purposes

        Parameters
        ----------
        texture : numpy.ndarray
            3-Dimensional numerical description of the dynamic texture (video
            data)

        Returns
        -------
        None.

        """

        _, axs = plt.subplots(self.segmentation_[0], self.segmentation_[1])

        if self.segmentation_[0] == self.segmentation_[1] \
           and self.segmentation_[1] == 1 :

            hist_xy, hist_yt, hist_xt = self.__generate_plane_histograms(texture)

            xy_vals = list(hist_xy.values())
            yt_vals = list(hist_yt.values())
            xt_vals = list(hist_xt.values())

            axs.bar(range(len(xy_vals)), xy_vals)
            axs.bar(range(len(xy_vals), len(yt_vals) + len(xy_vals)), yt_vals)
            axs.bar(range(len(yt_vals) + len(xy_vals), len(xt_vals) + len(yt_vals) + len(xy_vals)), yt_vals)

            return

        axs = axs.reshape(-1)
        for ax, segment in zip(axs, LBPTOPTool.__segment_texture(texture,self.segmentation_)):
            hist_xy, hist_yt, hist_xt = self.__generate_plane_histograms(segment)

            xy_vals = list(hist_xy.values())
            yt_vals = list(hist_yt.values())
            xt_vals = list(hist_xt.values())

            ax.bar(range(len(xy_vals)), xy_vals)
            ax.bar(range(len(xy_vals), len(yt_vals) + len(xy_vals)), yt_vals)
            ax.bar(range(len(yt_vals) + len(xy_vals), len(xt_vals) + len(yt_vals) + len(xy_vals)), yt_vals)

    def fit(self, train_samples, class_labels):
        """
        fit \:
            Fits the embedded KNN classifier object to the samples in 
            train_samples and class_labels

        Parameters
        ----------
        train_samples : numpy.ndarray
            Training set data samples
        class_labels : 
            Class labels that correspond to the training set data samples

        Returns
        -------
        self
            LBPTOPTool object with fitted classifier 

        """
        
        hists = self.transform(train_samples)
        self.classifier_.fit(hists, class_labels)

        return self

    def predict(self, samples, neighboors=1):
        """
        predict \:
            
        Predicts a samples class, or a set of samples classes using the
        previously fitted KNN classifier of the LBP tool

        Parameters
        ----------
        samples \: numpy.ndarray
            A single sample or a set of samples to be classified
            
        neighboors : int, optional, default = 1
            The number of neighboors to be used when using the KNN classifier

        Returns
        -------
        predictions
            The predicted class labels

        """

        hists = self.transform(samples)
        if len(hists.shape) == 2 :
            return self.classifier_.predict(hists, neighboors)

        predictions = list()
        for hist in hists :
            predictions.append(self.classifier_.predict(hist, neighboors))

        return predictions
