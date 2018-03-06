from __future__ import division
from math import log


class Node():
    '''
    This class is representation of a decision tree's branching points.
    '''
    def __init__(self):
        self.name = "" # Holds the feature to be breaking the dataset upon
        self.children = [] # Holds the one layer deeper in the node
        self.branches = [] # Holds the atrributes unique values in a feature
        self.path = "" # Records in string how we are able to reach that node. // we will record this from parent rows to child rows
        self.isLeaf = False # indicates if the decision has been made. Indicate if the tree is over as layer-wise

output = None # output features name
OUTPUTATTRIBUTES = None # output columns attributes
# // this global since these are main objectives we will use it a lot in order to be efficient we made it global



def Main():
    global output
    global OUTPUTATTRIBUTES

    dataSetTable = getDataSetTable('./dataSets/dataset1.txt') # reading the data

    # get data decision criterias
    output = dataSetTable[0][len(dataSetTable[0])-1] #
    OUTPUTATTRIBUTES = getAttributes(output, dataSetTable)

    # split the examples of the data with the features
    labels = dataSetTable[0]  # Features' names
    data = dataSetTable[1:][:]  # examples , occurunces, data

    ### begining of the algo. ###
    #starts the recursive function until it forms a Decision Tree
    root = constructDecisionTree(data, labels)

    # Displays the output for each path way. For example if the decision tree has 5 leaf it will display how it got there for each path
    displayTreePathWay(root)

def displayTreePathWay(node=None):
    '''
    This function outputs
    for each path way. For example if the decision tree has 5 leaf it will display how it got there for each path

    :param node  // the class of the decision tree on the highset level
    '''
    if node.isLeaf == True:
        # if this is leaf node first writh how it got there with recorded string on node.path & then write the decision
        print node.path,"OUTPUT =", node.name
    else:
        # if this not leaf write down to the nome how it got there and call this for every child
        for i in range (len(node.children)):
            # to each child node write the path how it got there
            node.children[i].path += str(node.path+node.name.upper()+"="+ node.branches[i]+"->")
            displayTreePathWay(node.children[i])

def constructDecisionTree(data,labels):
    '''
    :param data: this is the example set
    :param labels: this is the features names
    :return node: at the end returns a decision tree with nodes at different layers
    '''

    classList = [ex[-1] for ex in data]

    ### Control Phase starts

    # Control: if the column at hand has one attribute //or// entropy == 0 ?
    if classList.count(classList[0]) == len(classList):
        #returns any element of the class since they are the same
        return classList[0]

    # Control: Are we only left with a single example //or // entropy == 0 ?
    if len(data) == 1:
        return data[0][len(data[0])-1]
        ## Disclamer: We could have called entropy function but that would be inefficient since we can control with these two if statements

    # Control: Are we left with a single column
    if len(data[0]) == 1:
        # if we are left with just one column we can not extract any more information then we will return with the majority of the decision
        return majority(classList)

    ### Control Phase ends

    bestF = getBestFeature(data) # records the best feature we should use to divide out Data upon !-int (index)
    bestFLabel = labels[bestF]  # record the name of the feature in !-String

    node = Node() # Creates another node in the tree
    node.name = bestFLabel # labels the new class

    del(labels[bestF])
    featValues = [ex[bestF] for ex in data]
    uniqueVals = list(set(featValues))
    node.branches = uniqueVals

    # sets the attribute in the way to one layer deeper in the tree
    node.branches = list(set(featValues))

    for value in uniqueVals:
        subLabels = labels[:]
        newNode = constructDecisionTree(split(data, bestF, value),subLabels)
        if type(newNode) != str:
            node.children.append(newNode)
        else:
            n = Node()
            n.name = newNode
            n.isLeaf=True
            node.children.append(n)
    # when the code comes this point it constructed a complete tree in a class family
    return node



### MATH FUNCTIONS ###

def gain(parentEnt, entropyList, rowCount):
    '''
    Calculates the information gain of the given for each attribute

    :param parentEnt: the entropy of the higher layer
    :param entropyList: the entropies of the classes if we were to divide them
    :param rowCount:
    :return:
    '''
    weightedAverage = 0
    for i in range(len(entropyList[0])):
        weightedAverage += entropyList[1][i] / rowCount * entropyList[0][i]

    #print "gain:",float(format(parentEnt - weightedAverage, '.3f'))
    return float(format(parentEnt - weightedAverage, '.3f'))

def entropy(table):
    '''
    Calculates the entropy level by using the formula
    :param table:
    :return:
    '''
    array = [0 for i in range(len(OUTPUTATTRIBUTES))]


    for i in range(len(table)):
        current = table[i][len(table[0])-1]
        index = OUTPUTATTRIBUTES.index(current)
        array[index]+=1

    total = 0
    for i in array:
        i = i/sum(array)
        if i != 0:
            total += i * log(i,2)
    total *= -1
    total = float(format(total, '.3f'))

    return total, len(table)


### UTILITIES ###

def majority(classList):
    '''
    finds the majority decision in a single left feature table

    :param classList:
    :return:
    '''
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def getAttributes(feature, dataSetTable):
    '''
    returns the unique values in a feature column

    :param feature:
    :param dataSetTable:
    :return:
    '''
    attributes = []
    column = dataSetTable[0].index(feature)
    for i in range(1, len(dataSetTable)):
        if attributes.__contains__(dataSetTable[i][column]) is not True:
            attributes.append(dataSetTable[i][column])
    return attributes

def getDataSetTable(path):
    '''

    :param path:
    :return:
    '''
    table = []
    with open(path, 'r') as file:
        for line in file:
            tmp = []
            for word in line.split(","):
                tmp.append(word.strip())
            table.append(tmp)
    return table

def split(data, axis, val):
    '''
    splits the data into smaller chunks to prepare for recursive call

    new data will be selected of the rows of the attrbute that we are concerned but with out the feature

    :param data: example set
    :param axis: the column/feature to be deleted
    :param val:   the rows/attribute to be selected
    :return: # newdata
    '''

    newData = []
    for feat in data:
        if feat[axis] == val:
            reducedFeat = feat[:axis]
            reducedFeat.extend(feat[axis + 1:])
            newData.append(reducedFeat)
    return newData

def getBestFeature(data):
    '''
    finds the best feature we should divide our tree upon

    :param data:
    :return:
    '''
    features = len(data[0]) - 1

    informationGains = [0 for i in range(len(data[0]) - 1)]

    for i in range(features):
        parentEntropy, totalRow = entropy(data)

        attributes = getAttributes(data[0][i], data)
        childEntropies = [[0 for k in range(len(attributes))] for l in range(2)]
        counter = 0
        for j in attributes:  # for each attribute
            filteredTable = split(data, i, j)
            #print j
            #for k in filteredTable:
                #print k
            childEntropies[0][counter], childEntropies[1][counter] = entropy(filteredTable)
            counter += 1

        informationGains[i] = gain(parentEntropy, childEntropies, totalRow)
    return informationGains.index(max(informationGains))


if __name__ == '__main__':
    Main()