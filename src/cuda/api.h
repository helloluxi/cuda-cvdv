#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque handle — callers hold CVDVContext* but cannot inspect fields.
// Guard prevents redefinition when types.h (full struct) is already included.
#ifndef CVDV_TYPES_INCLUDED
typedef struct CVDVContext CVDVContext;
#endif

CVDVContext* cvdvCreate(int numReg, int* numQubits);
void         cvdvDestroy(CVDVContext* ctx);
void         cvdvFree(CVDVContext* ctx);
void         cvdvInitFromSeparable(CVDVContext* ctx, void** devicePtrs, int numReg);
void         cvdvSetStateFromDevicePtr(CVDVContext* ctx, void* d_src);
int          cvdvGetNumRegisters(CVDVContext* ctx);
size_t       cvdvGetTotalSize(CVDVContext* ctx);
void         cvdvGetRegisterInfo(CVDVContext* ctx, int* qubitCountsOut, double* gridStepsOut);
int          cvdvGetRegisterDim(CVDVContext* ctx, int regIdx);
double       cvdvGetRegisterDx(CVDVContext* ctx, int regIdx);

void   cvdvFtQ2P(CVDVContext* ctx, int regIdx);
void   cvdvFtP2Q(CVDVContext* ctx, int regIdx);
void   cvdvDisplacement(CVDVContext* ctx, int regIdx, double betaRe, double betaIm);
void   cvdvConditionalDisplacement(CVDVContext* ctx, int targetReg, int ctrlReg,
                                    int ctrlQubit, double betaRe, double betaIm);
void   cvdvPauliRotation(CVDVContext* ctx, int regIdx, int qubitIdx, int axis, double theta);
void   cvdvHadamard(CVDVContext* ctx, int regIdx, int qubitIdx);
void   cvdvParity(CVDVContext* ctx, int regIdx);
void   cvdvConditionalParity(CVDVContext* ctx, int targetReg, int ctrlReg, int ctrlQubit);
void   cvdvSwapRegisters(CVDVContext* ctx, int reg1, int reg2);
void   cvdvPhaseSquare(CVDVContext* ctx, int regIdx, double t);
void   cvdvPhaseCubic(CVDVContext* ctx, int regIdx, double t);
void   cvdvRotation(CVDVContext* ctx, int regIdx, double theta);
void   cvdvConditionalRotation(CVDVContext* ctx, int targetReg, int ctrlReg,
                                int ctrlQubit, double theta);
void   cvdvSqueeze(CVDVContext* ctx, int regIdx, double r);
void   cvdvConditionalSqueeze(CVDVContext* ctx, int targetReg, int ctrlReg,
                               int ctrlQubit, double r);
void   cvdvBeamSplitter(CVDVContext* ctx, int reg1, int reg2, double theta);
void   cvdvConditionalBeamSplitter(CVDVContext* ctx, int reg1, int reg2,
                                    int ctrlReg, int ctrlQubit, double theta);
void   cvdvQ1Q2Gate(CVDVContext* ctx, int reg1, int reg2, double coeff);

void   cvdvGetWigner(CVDVContext* ctx, int regIdx, double* wignerOut);
void   cvdvGetHusimiQ(CVDVContext* ctx, int regIdx, double* husimiOut);
void   cvdvGetHusimiQOverlap(CVDVContext* ctx, int regIdx, double* husimiOut);
void   cvdvGetHusimiQWigner(CVDVContext* ctx, int regIdx, double* husimiOut);

void   cvdvMeasureMultiple(CVDVContext* ctx, const int* regIdxs, int numRegs,
                            double* probsOut);
void   cvdvMeasureMultipleCT(CVDVContext* ctx, const int* regIdxs, int numRegs,
                              double* probsOut);
void   cvdvGetState(CVDVContext* ctx, double* realOut, double* imagOut);
double cvdvGetNorm(CVDVContext* ctx);
void   cvdvGetFidelity(CVDVContext* ctx, void** devicePtrs, int numReg, double* fidOut);
double cvdvGetPhotonNumber(CVDVContext* ctx, int regIdx);
void   cvdvFidelityStatevectors(CVDVContext* ctx1, CVDVContext* ctx2, double* fidOut);

#ifdef __cplusplus
}
#endif
