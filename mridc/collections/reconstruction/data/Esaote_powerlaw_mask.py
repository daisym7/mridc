import numpy as np
import matplotlib.pyplot as plt

'''LB: this function applies an offset(raising the function) it should be the responsibility 
of this function to create the central plateau, saturating the probability to 1 near the center'''
def OffsetPDF(pdf, fOffset):
    pdf = pdf + fOffset
    return pdf


def TruncatePDF(pdf, fLimit):
    return np.where(pdf > fLimit, fLimit, pdf)


def GetFloorPDFSum(pIn):
    fTmp = np.sum(pIn)
    fSum = np.floor(fTmp)
    return fSum


def GenPDFOri(r, p):
    return (1-r)**p


def GenPDF(imSize, p, PCTG):
    # initialize pdf
    r = np.abs(np.linspace(-1.0, 1.0, imSize))
    pdfOrig = GenPDFOri(r, p)

    # get min and max of original pdf for the offset
    minval = np.min(pdfOrig)
    maxval = np.max(pdfOrig)

    # find right pdf
    counter = 0
    while True:
        # calculate offset
        val = (minval / 2.0) + (maxval / 2.0)
        # add offset to original pdf
        pdf = OffsetPDF(pdfOrig, val)
        # truncate pdf values to 1
        pdf = TruncatePDF(pdf, 1.0)
        # get sum of all pdf values
        N = GetFloorPDFSum(pdf)

        # adjust to get different offset
        if N > PCTG:
            maxval = val
        elif N < PCTG:
            minval = val
        # break if sum of pdf is equal to wanted lines
        elif N == PCTG:
            break

        # To not get stuck in an infinite while loop
        counter += 1
        if counter > 10000:
            raise ValueError(f"Cannot generate mask.")

    return pdf


def Calc_AsymCond(pIn, fThreshold, PCTG):
    nSize = len(pIn)
    center_point = int(nSize / 2)

    nSumLeft = np.sum(pIn[:center_point])
    nSumRight = np.sum(pIn[center_point:])

    if fThreshold == 0.0:
        if (PCTG % 2 == 1) and (nSize % 2 == 0):
            fThreshold = 1.0
        if (PCTG % 2 == 0) and (nSize % 2 == 1) and (pIn[nSize/2] == 1.0):
            fThreshold = 1.0

    condition = abs(nSumLeft - nSumRight) <= fThreshold
    return condition


def Calc_EncNumbTol(pIn, fThreshold, fTollerance):
    fSum = np.sum(pIn)
    EncNumDiff = abs(fSum - fThreshold)
    condition = EncNumDiff <= fTollerance
    return condition


def GenMask(pdf, iter, tol, PCTG, AsymThresh):
    # print("Vector size", len(pdf), PCTG)
    np.random.seed()

    tmp = np.zeros(len(pdf))

    minIntr = 1000000000

    for n in range(1, iter):
        bFound = False
        for j in range(100000):
            fTmp = np.random.rand(len(pdf))
            tmp = np.where(fTmp <= pdf, 1, 0)

            # check if the correct number of lines (PCTG) are kept
            EncNumbTol_Cond = Calc_EncNumbTol(tmp, PCTG, tol)
            # check if left and right of the center are the same amount of lines
            Asym_Cond = Calc_AsymCond(tmp, AsymThresh, PCTG)

            if EncNumbTol_Cond and Asym_Cond:
                bFound = True
                break

        if not bFound:
            print("Impossibile soddisfare le condizioni sulla maschera.")
            raise ValueError(f"Cannot generate mask.")


        # Point spread Function
        ratio = tmp/pdf
        log2 = int(np.ceil(np.log(len(pdf))/np.log(2)))
        nSize = int(2 ** log2)
        pDst = np.zeros(nSize)
        pDst[:len(ratio)] = ratio
        pDst = pDst.astype(complex)
        # take FT to see clustering of lines
        pDst = np.fft.ifft(pDst)
        pMag = np.abs(pDst)
        # set first line (the centerlines) to 0
        pMag[0] = 0
        pMag = pMag[:-1]
        max_mag = np.max(pMag)
        if max_mag < minIntr:
            minIntr = max_mag
            minIntrVec = tmp

    return minIntrVec

if __name__ == "__main__":
    # Example usage:
    shape = (180, 180)
    acc = 2
    number_lines = 90
    AsymTolerance = 0.1
    iterations = 1000
    tolerance = 0.01
    PDF = []
    MASK = []


    # mask_vector = GenMask(pdf, iterations, tolerance, number_lines, AsymTolerance)
    # mask6 = np.ones((shape[0], shape[1])) * np.array(mask_vector)
    # plt.imshow(mask6, cmap='gray')
    # plt.show()



