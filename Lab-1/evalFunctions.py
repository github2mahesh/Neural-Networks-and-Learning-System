import numpy as np

def calcAccuracy(LPred, LTrue):
    """Calculates prediction accuracy from data labels.

    Args:
        LPred (array): Predicted data labels.
        LTrue (array): Ground truth data labels.

    Retruns:
        acc (float): Prediction accuracy.
    """

    # --------------------------------------------
    count=0
    for i in range(len(LPred)):
        if LPred[i]==LTrue[i]:
            count+=1
    # --------------------------------------------
    acc = count/len(LPred)
    # ============================================
    return acc


def calcConfusionMatrix(LPred, LTrue):
    """Calculates a confusion matrix from data labels.

    Args:
        LPred (array): Predicted data labels.
        LTrue (array): Ground truth data labels.

    Returns:
        cM (array): Confusion matrix, with predicted labels in the rows
            and actual labels in the columns.
    """

    # --------------------------------------------
    l=np.unique(LTrue)
    cM=np.zeros((len(l),len(l)))
    for i in l:
        for j in range(len(LTrue)):
            if LTrue[j]==i:
                k=np.where(l==i)[0][0]
                n=np.where(l==LPred[j])[0][0]
                cM[k,n]+=1
    # --------------------------------------------
    
    # ============================================

    return cM


def calcAccuracyCM(cM):
    """Calculates prediction accuracy from a confusion matrix.

    Args:
        cM (array): Confusion matrix, with predicted labels in the rows
            and actual labels in the columns.

    Returns:
        acc (float): Prediction accuracy.
    """

    # --------------------------------------------
    acc=np.trace(cM)/np.sum(cM)
    # --------------------------------------------
    
    # ============================================
    
    return acc
