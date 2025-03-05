// @HEADER
// *****************************************************************************
//        MueLu: A package for multigrid based preconditioning
//
// Copyright 2012 NTESS and the MueLu contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

// MueLu
#include <MueLu_CreateXpetraPreconditioner.hpp>
#include <MueLu_ConfigDefs.hpp>
#include <MueLu_ParameterListInterpreter.hpp>

// Teuchos
#include <Teuchos_XMLParameterListHelpers.hpp>

// Belos
#include <BelosPseudoBlockGmresSolMgr.hpp>
#include <BelosStatusTestCombo.hpp>
#include <BelosXpetraStatusTestGenResSubNorm.hpp>
#include <BelosXpetraAdapter.hpp>
#include <BelosMueLuAdapter.hpp>

template <typename GlobalOrdinal>
void read_Lagr2Dof(std::string filemane, std::map<GlobalOrdinal, GlobalOrdinal> &lagr2Dof) {
  std::fstream lagr2DofFile;
  lagr2DofFile.open(filemane);
  TEUCHOS_ASSERT(lagr2DofFile.is_open())

  GlobalOrdinal key;
  GlobalOrdinal value;
  while (lagr2DofFile >> key >> value) {
    lagr2Dof[key] = value;
  }
  lagr2DofFile.close();
}

bool hasLocalValueAt(const Teuchos::RCP<const Xpetra::Matrix<double, int, int>>& matrix,
  int localRow, int localCol) {
Teuchos::Array<int> indices;
Teuchos::Array<double> values;
size_t numEntries = matrix->getNumEntriesInLocalRow(localRow);

if (numEntries == 0) return false;

indices.resize(numEntries);
values.resize(numEntries);

matrix->getLocalRowCopy(localRow, indices(), values(), numEntries);

for (size_t i = 0; i < numEntries; ++i) {
  std::cout<<"indices["<<i<<"] = "<<indices[i]<<"\n";
if (indices[i] == localCol) return true;
}
return false;
}

double getValueAt(const Teuchos::RCP<const Xpetra::Matrix<double, int, int>>& matrix,
  int row, int col) {
Teuchos::Array<int> indices;
Teuchos::Array<double> values;
size_t numEntries = matrix->getNumEntriesInGlobalRow(row);

if (numEntries == 0) return 0.0; // No nonzero entries in this row

indices.resize(numEntries);
values.resize(numEntries);

// Extract row data
matrix->getLocalRowCopy(row, indices(), values(), numEntries);

// Find the column
for (size_t i = 0; i < numEntries; ++i) {
if (indices[i] == col) {
return values[i]; // Return value at (row, col)
}
}
return 0.0; // If column not found, assume zero (sparse matrix)
}




template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
int main_(Teuchos::CommandLineProcessor &clp, Xpetra::UnderlyingLib &lib, int argc, char *argv[]) {
#include <MueLu_UseShortNames.hpp>

  // The MeshTyingBlocked_NodeBased tests only work with real Scalar types,
  if (Teuchos::ScalarTraits<Scalar>::isComplex) return EXIT_SUCCESS;

  using SparseMatrixType    = Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using tpetra_mvector_type = Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using tpetra_map_type     = Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>;

  using Teuchos::RCP;
  using Teuchos::rcp;
  using namespace Teuchos;

  using ST = ScalarTraits<Scalar>;

  oblackholestream blackhole;

  RCP<const Comm<int>> comm = DefaultComm<int>::getComm();
  RCP<FancyOStream> out     = fancyOStream(rcpFromRef(std::cout));
  out->setOutputToRootOnly(0);

  const GO globalPrimalNumDofs    = 1599;
  const GO globalDualNumDofs      = 300;
  const GO globalNumDofs          = globalPrimalNumDofs + globalDualNumDofs;  // used for the maps
  const size_t nPrimalDofsPerNode = 3;
  const GO globalPrimalNumNodes   = globalPrimalNumDofs / nPrimalDofsPerNode;
  const size_t nDualDofsPerNode   = 3;

  std::map<GO, GO> lagr2Dof;
  std::map<LO, LO> myLagr2Dof;
  read_Lagr2Dof<GO>("Lagr2Dof.txt", lagr2Dof);

  // Construct the blocked map using the Xpetra-style indexing
  RCP<const tpetra_map_type> primalNodeMap = Tpetra::createUniformContigMapWithNode<LocalOrdinal, GlobalOrdinal, Node>(globalPrimalNumNodes, comm);
  const GO indexBase                       = primalNodeMap->getIndexBase();
  ArrayView<const GO> myPrimalNodes        = primalNodeMap->getLocalElementList();

  const size_t numMyPrimalNodes = primalNodeMap->getLocalNumElements();
  const size_t numMyPrimalDofs  = numMyPrimalNodes * nPrimalDofsPerNode;

  Array<GO> myPrimalDofs(numMyPrimalDofs);

  LO current_i = 0;
  for (size_t i = 0; i < numMyPrimalNodes; ++i)
    for (size_t j = 0; j < nPrimalDofsPerNode; ++j)
      myPrimalDofs[current_i++] = myPrimalNodes[i] * nPrimalDofsPerNode + j;

  RCP<const tpetra_map_type> primalMap = rcp(new tpetra_map_type(globalPrimalNumDofs, myPrimalDofs, indexBase, comm));

  size_t numMyDualDofs = 0;

  for (auto i = lagr2Dof.begin(); i != lagr2Dof.end(); ++i)
    if (primalMap->isNodeGlobalElement(nPrimalDofsPerNode * (i->second)))
      ++numMyDualDofs;

  numMyDualDofs *= nDualDofsPerNode;

  std::cout<<"test line 98: numMyDualDofs = "<<numMyDualDofs<<"\n";

  const size_t numMyDofs = numMyPrimalDofs + numMyDualDofs;

  Array<GO> myDualDofs(numMyDualDofs);
  Array<GO> myDualDofs2(numMyDualDofs);
  Array<GO> myDofs(numMyDofs);

  for (size_t i = 0; i < numMyPrimalDofs; ++i)
    myDofs[i] = myPrimalDofs[i];

  current_i = 0;
  for (auto i = lagr2Dof.begin(); i != lagr2Dof.end(); ++i)
    if (primalMap->isNodeGlobalElement(nPrimalDofsPerNode * (i->second))) {
      for (size_t j = 0; j < nDualDofsPerNode; ++j) {
        myDualDofs[nDualDofsPerNode * current_i + j]               = (i->first) * nDualDofsPerNode + j;
        myDualDofs2[nDualDofsPerNode * current_i + j]               = (i->first) * nDualDofsPerNode + j + globalPrimalNumDofs;
        myDofs[numMyPrimalDofs + nDualDofsPerNode * current_i + j] = globalPrimalNumDofs + (i->first) * nDualDofsPerNode + j;
      }
      GO primalDof          = nPrimalDofsPerNode * (i->second);
      myLagr2Dof[current_i] = primalMap->getLocalElement(primalDof) / nPrimalDofsPerNode;
      ++current_i;
    }

  RCP<const tpetra_map_type> dualMap = rcp(new tpetra_map_type(globalDualNumDofs, myDualDofs, indexBase, comm));
  RCP<const tpetra_map_type> dualMap2 = rcp(new tpetra_map_type(globalDualNumDofs, myDualDofs2, indexBase, comm));
  std::cout<<"test line 121: dualMap = \n";
  dualMap->describe(*Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout)), Teuchos::VERB_EXTREME);
  std::cout<<"test line 126: dualMap2 = \n";
  dualMap2->describe(*Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout)), Teuchos::VERB_EXTREME);
  std::cout<<"test line 128: primalMap = \n";
  primalMap->describe(*Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout)), Teuchos::VERB_EXTREME);

  RCP<const tpetra_map_type> fullMap = rcp(new tpetra_map_type(globalNumDofs, myDofs, indexBase, comm));

  std::cout<<"test line 128: fullMap =\n";
  fullMap->describe(*Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout)), Teuchos::VERB_EXTREME);

  RCP<const Map> fullXMap   = rcp(new TpetraMap(fullMap));
  RCP<const Map> primalXMap = rcp(new TpetraMap(primalMap));
  RCP<const Map> dualXMap   = rcp(new TpetraMap(dualMap));
  RCP<const Map> dualXMap2   = rcp(new TpetraMap(dualMap2));

  std::vector<RCP<const Map>> xsubmaps = {primalXMap, dualXMap2};
  RCP<BlockedMap> blockedMap           = rcp(new BlockedMap(fullXMap, xsubmaps, false));

  // Read input matrices
  typedef Tpetra::MatrixMarket::Reader<SparseMatrixType> reader_type;

  RCP<Matrix> xQ  = Xpetra::IO<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Read("Q_mm.txt", primalXMap, primalXMap, primalXMap, primalXMap);
  RCP<Matrix> xG  = Xpetra::IO<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Read("G_3dof_mm.txt", primalXMap, dualXMap, dualXMap, primalXMap);
  std::cout<<"test line 149: xG->getColMap()->describe() =\n";
  xG->getColMap()->describe(*Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout)), Teuchos::VERB_EXTREME);

  RCP<Matrix> xGT = Xpetra::IO<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Read("GT_3dof_mm.txt", dualXMap, primalXMap, primalXMap, dualXMap);
  RCP<Matrix> xC  = Xpetra::IO<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Read("C_3dof_mm.txt", dualXMap, dualXMap, dualXMap, dualXMap);
  ////////////

std::cout<<"xG\n";
/*
    const GO globalPrimalNumDofs    = 1599;
  const GO globalDualNumDofs      = 300;
  const GO globalNumDofs          = globalPrimalNumDofs + globalDualNumDofs;  // used for the maps
  const size_t nPrimalDofsPerNode = 3;
  const GO globalPrimalNumNodes   = globalPrimalNumDofs / nPrimalDofsPerNode;
  const size_t nDualDofsPerNode   = 3;
    */



    /*Teuchos::Array<GlobalOrdinal> GRows;
    for(int i=0; i<globalPrimalNumDofs; i++){
      size_t numEntries = xG->getNumEntriesInLocalRow(i);

      if (numEntries != 0) GRows.push_back(i);
    }

    Teuchos::RCP<const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node>> GrowMap =
        Xpetra::MapFactory<LocalOrdinal, GlobalOrdinal, Node>::Build(
            Xpetra::UseTpetra, GRows.size(), GRows(), 0, comm);*/

            Teuchos::Array<GlobalOrdinal> GCols;
    for(int i=0; i<globalPrimalNumDofs; i++){
      size_t numEntries = xG->getNumEntriesInLocalRow(i);

      if (numEntries != 0) GCols.push_back(i);
    }

    Teuchos::RCP<const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node>> GrowMap =
        Xpetra::MapFactory<LocalOrdinal, GlobalOrdinal, Node>::Build(
            Xpetra::UseTpetra, GCols.size(), GCols(), 0, comm);



    


      ////RCP<Matrix> xG_new = Xpetra::MatrixFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Build(GrowMap, xG->getGlobalMaxNumRowEntries());
      RCP<Matrix> xG_newNotWrapped = Xpetra::MatrixFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Build(primalXMap, dualXMap2, xG->getGlobalMaxNumRowEntries());
      RCP<CrsMatrixWrap> xG_new = Teuchos::rcp_dynamic_cast<CrsMatrixWrap>(xG_newNotWrapped);
{
    for(int i=0; i<globalPrimalNumDofs; i++){
      Teuchos::Array<int> indices;
      Teuchos::Array<double> values;
      

      if (xG->getNumEntriesInLocalRow(i) <= 0) continue;

      size_t numEntries = xG->getNumEntriesInLocalRow(i);

      indices.resize(numEntries);
      values.resize(numEntries);

      xG->getLocalRowCopy(i, indices(), values(), numEntries);

      for (size_t j = 0; j < numEntries; ++j) {
        std::cout<<"indices["<<j<<"] = "<<indices[j]<<"\n";

        Teuchos::Array<GlobalOrdinal> colIndices2(1, indices[j]+globalPrimalNumDofs); // Column index
        Teuchos::Array<Scalar> values2(1, getValueAt(xG, i, indices[j]));          // Corresponding value
        xG_new->insertGlobalValues(i, colIndices2(), values2());
      }
    }
//xG_new->replaceColumnMap(dualXMap2);

    xG_new->fillComplete(dualXMap2, primalXMap);
  }

    /*
    const GO globalPrimalNumDofs    = 1599;
  const GO globalDualNumDofs      = 300;
  const GO globalNumDofs          = globalPrimalNumDofs + globalDualNumDofs;  // used for the maps
  const size_t nPrimalDofsPerNode = 3;
  const GO globalPrimalNumNodes   = globalPrimalNumDofs / nPrimalDofsPerNode;
  const size_t nDualDofsPerNode   = 3;
*/
    // xGT
    std::cout<<"xGT\n";

    Teuchos::Array<GlobalOrdinal> GTRows;
    for(int i=0; i<globalPrimalNumDofs; i++){
      GlobalOrdinal numEntries = xGT->getNumEntriesInLocalRow(i);

      if (numEntries > 0){
        GTRows.push_back(i+globalPrimalNumDofs);
      }
    }

    Teuchos::RCP<const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node>> GTrowMap =
        Xpetra::MapFactory<LocalOrdinal, GlobalOrdinal, Node>::Build(
            Xpetra::UseTpetra, GTRows.size(), GTRows(), 0, comm);

            std::cout<<"test line 279: GTrowMap->describe() =\n";
            GTrowMap->describe(*Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout)), Teuchos::VERB_EXTREME);


    ////xGT_new = Xpetra::MatrixFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Build(GTrowMap, GrowMap, xGT->getGlobalMaxNumRowEntries());
    RCP<Matrix> xGT_newNotWrapped = Xpetra::MatrixFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Build(dualXMap2, primalXMap, xGT->getGlobalMaxNumRowEntries());
    RCP<CrsMatrixWrap> xGT_new = Teuchos::rcp_dynamic_cast<CrsMatrixWrap>(xGT_newNotWrapped);
{
    for(int i=0; i<globalDualNumDofs; i++){
      Teuchos::Array<int> indices;
      Teuchos::Array<double> values;

      if (xGT->getNumEntriesInLocalRow(i) <= 0) continue;

      size_t numEntries = xGT->getNumEntriesInLocalRow(i);

      indices.resize(numEntries);
      values.resize(numEntries);

      xGT->getLocalRowCopy(i, indices(), values(), numEntries);

      for (size_t j = 0; j < numEntries; ++j) {

        Teuchos::Array<GlobalOrdinal> colIndices2(1, indices[j]); // Column index
        Teuchos::Array<Scalar> values2(1, getValueAt(xGT, i, indices[j]));          // Corresponding value
        xGT_new->insertGlobalValues(i+globalPrimalNumDofs, colIndices2(), values2());
      }
    }
    //xGT_new->replaceColumnMap(primalXMap);

    

    xGT_new->fillComplete(primalXMap, dualXMap2);
  }

  




std::cout<<"xC\n";

  // xC
    

  Teuchos::Array<GlobalOrdinal> CRows;
    for(int i=0; i<globalDualNumDofs; i++){
      size_t numEntries = xC->getNumEntriesInLocalRow(i);

      if (numEntries > 0) CRows.push_back(i+globalPrimalNumDofs);
    }

    Teuchos::RCP<const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node>> CrowMap =
        Xpetra::MapFactory<LocalOrdinal, GlobalOrdinal, Node>::Build(
            Xpetra::UseTpetra, CRows.size(), CRows(), 0, comm);


    

  ////xC_new = Xpetra::MatrixFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Build(CrowMap, xC->getGlobalMaxNumRowEntries());
  RCP<Matrix> xC_newNotWrapped = Xpetra::MatrixFactory<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Build(dualXMap2, dualXMap2, xC->getGlobalMaxNumRowEntries());
  RCP<CrsMatrixWrap> xC_new = Teuchos::rcp_dynamic_cast<CrsMatrixWrap>(xC_newNotWrapped);
{
  for(int i=0; i<globalDualNumDofs; i++){
    Teuchos::Array<int> indices;
    Teuchos::Array<double> values;
    

    if (xC->getNumEntriesInLocalRow(i) <= 0) continue;

    size_t numEntries = xC->getNumEntriesInLocalRow(i);

    indices.resize(numEntries);
    values.resize(numEntries);

    xC->getLocalRowCopy(i, indices(), values(), numEntries);

    for (size_t j = 0; j < numEntries; ++j) {

      Teuchos::Array<GlobalOrdinal> colIndices2(1, indices[j]+globalPrimalNumDofs); // Column index
      Teuchos::Array<Scalar> values2(1, getValueAt(xC, i, indices[j]));          // Corresponding value
      xC_new->insertGlobalValues(i+globalPrimalNumDofs, colIndices2(), values2());
    }
  }
  //xC_new->replaceColumnMap(dualXMap2);

  xC_new->fillComplete(dualXMap2, dualXMap2);
}


















  std::cout<<"test line 401: xG_new->getColMap()->describe() =\n";
  xG_new->getColMap()->describe(*Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout)), Teuchos::VERB_EXTREME);

  std::cout<<"test line 404: xG_new->getRowMap()->describe() =\n";
  xG_new->getRowMap()->describe(*Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout)), Teuchos::VERB_EXTREME);

  std::cout<<"test line 407: xG_new->getDomainMap()->describe() =\n";
  xG_new->getDomainMap()->describe(*Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout)), Teuchos::VERB_EXTREME);

  std::cout<<"test line 410: xG_new->getRangeMap()->describe() =\n";
  xG_new->getRangeMap()->describe(*Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout)), Teuchos::VERB_EXTREME);


  std::cout<<"test line 414: xGT_new->getColMap()->describe() =\n";
  xGT_new->getColMap()->describe(*Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout)), Teuchos::VERB_EXTREME);

  std::cout<<"test line 417: xGT_new->getRowMap()->describe() =\n";
  xGT_new->getRowMap()->describe(*Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout)), Teuchos::VERB_EXTREME);

  std::cout<<"test line 420: xGT_new->getDomainMap()->describe() =\n";
  xGT_new->getDomainMap()->describe(*Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout)), Teuchos::VERB_EXTREME);

  std::cout<<"test line 423: xGT_new->getRangeMap()->describe() =\n";
  xGT_new->getRangeMap()->describe(*Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout)), Teuchos::VERB_EXTREME);


  std::cout<<"test line 427: xC_new->getColMap()->describe() =\n";
  xC_new->getColMap()->describe(*Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout)), Teuchos::VERB_EXTREME);

  std::cout<<"test line 430: xC_new->getRowMap()->describe() =\n";
  xC_new->getRowMap()->describe(*Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout)), Teuchos::VERB_EXTREME);

  std::cout<<"test line 433: xC_new->getDomainMap()->describe() =\n";
  xC_new->getDomainMap()->describe(*Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout)), Teuchos::VERB_EXTREME);

  std::cout<<"test line 436: xC_new->getRangeMap()->describe() =\n";
  xC_new->getRangeMap()->describe(*Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout)), Teuchos::VERB_EXTREME);

//////////////////////////////////////////


  RCP<CrsMatrixWrap> xwQ  = Teuchos::rcp_dynamic_cast<CrsMatrixWrap>(xQ);
  RCP<CrsMatrixWrap> xwG  = Teuchos::rcp_dynamic_cast<CrsMatrixWrap>(xG_new);
  RCP<CrsMatrixWrap> xwGT = Teuchos::rcp_dynamic_cast<CrsMatrixWrap>(xGT_new);
  RCP<CrsMatrixWrap> xwC  = Teuchos::rcp_dynamic_cast<CrsMatrixWrap>(xC_new);

  /*Teuchos::RCP<Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>> tpetraMat =
            Xpetra::Helpers<Scalar, LocalOrdinal, GlobalOrdinal, Node>::Op2TpetraCrs(xwG->getCrsMatrix());

  tpetraMat->replaceColMap(dualXMap2);

  std::cout<<"test line 376: (xwG->getCrsMatrix())->getColMap()->describe() =\n";
  (xwG->getCrsMatrix())->getColMap()->describe(*Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout)), Teuchos::VERB_EXTREME);

  std::cout<<"test line 379: (xwG->getCrsMatrix())->getRowMap()->describe() =\n";
  (xwG->getCrsMatrix())->getRowMap()->describe(*Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout)), Teuchos::VERB_EXTREME);*/

  //(xwG->getCrsMatrix())->replaceColumnMap(dualXMap2);
  //(xwGT->getCrsMatrix())->replaceColumnMap(primalXMap);
  //(xwC->getCrsMatrix())->replaceColumnMap(dualXMap2);





  RCP<BlockedCrsMatrix> blockedMatrix = rcp(new BlockedCrsMatrix(blockedMap, blockedMap, 8));

  blockedMatrix->setMatrix(0, 0, xwQ);
  blockedMatrix->setMatrix(0, 1, xwG);
  blockedMatrix->setMatrix(1, 0, xwGT);
  blockedMatrix->setMatrix(1, 1, xwC);
  blockedMatrix->fillComplete();

  std::cout<<"domainMap of blockedMatrix(0,1):\n";
  ((blockedMatrix->getMatrix(0, 1))->getDomainMap())->describe(*Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout)), Teuchos::VERB_EXTREME);

  std::cout<<"rangeMap of blockedMatrix(0,1):\n";
  ((blockedMatrix->getMatrix(0, 1))->getRangeMap())->describe(*Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout)), Teuchos::VERB_EXTREME);

  std::cout<<"colMap of blockedMatrix(0,1):\n";
  ((blockedMatrix->getMatrix(0, 1))->getColMap())->describe(*Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout)), Teuchos::VERB_EXTREME);

  std::cout<<"rowMap of blockedMatrix(0,1):\n";
  ((blockedMatrix->getMatrix(0, 1))->getRowMap())->describe(*Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout)), Teuchos::VERB_EXTREME);



  std::cout<<"domainMap of blockedMatrix(1,0):\n";
  ((blockedMatrix->getMatrix(1, 0))->getDomainMap())->describe(*Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout)), Teuchos::VERB_EXTREME);

  std::cout<<"rangeMap of blockedMatrix(1,0):\n";
  ((blockedMatrix->getMatrix(1, 0))->getRangeMap())->describe(*Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout)), Teuchos::VERB_EXTREME);

  std::cout<<"colMap of blockedMatrix(1,0):\n";
  ((blockedMatrix->getMatrix(1, 0))->getColMap())->describe(*Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout)), Teuchos::VERB_EXTREME);

  std::cout<<"rowMap of blockedMatrix(1,0):\n";
  ((blockedMatrix->getMatrix(1, 0))->getRowMap())->describe(*Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout)), Teuchos::VERB_EXTREME);






  

  // Create the preconditioner
  std::string xmlFile = "simple_3dof.xml";

  RCP<ParameterList> params     = Teuchos::getParametersFromXmlFile(xmlFile);
  ParameterList &userDataParams = params->sublist("user data");
  userDataParams.set<RCP<std::map<LO, LO>>>("DualNodeID2PrimalNodeID", rcpFromRef(myLagr2Dof));

  RCP<Hierarchy> H = MueLu::CreateXpetraPreconditioner(Teuchos::rcp_dynamic_cast<Matrix>(blockedMatrix, true), *params);
  ///RCP<MueLu::Hierarchy<SC, LO, GO, NO>> H_ = mueLuFactory.CreateHierarchy();
  H->IsPreconditioner(true);
  H->GetLevel(0)->Set(
    "A", Teuchos::rcp_dynamic_cast<Xpetra::Matrix<SC, LO, GO, NO>>(blockedMatrix));

  // Create the preconditioned GMRES solver
  typedef typename tpetra_mvector_type::dot_type belos_scalar;
  typedef Belos::OperatorT<MultiVector> OP;

  typedef Belos::StatusTestGenResSubNorm<belos_scalar, MultiVector, OP> blockStatusTestClass;
  typedef Belos::StatusTestCombo<belos_scalar, MultiVector, OP> StatusTestComboClass;

  typename ST::magnitudeType tol  = 1e-4;
  typename ST::magnitudeType bTol = 1e-5;

  RCP<blockStatusTestClass> primalBlockStatusTest = rcp(new blockStatusTestClass(bTol, 0));
  RCP<blockStatusTestClass> dualBlockStatusTest   = rcp(new blockStatusTestClass(bTol, 1));

  RCP<StatusTestComboClass> statusTestCombo = rcp(new StatusTestComboClass(StatusTestComboClass::SEQ));
  statusTestCombo->addStatusTest(primalBlockStatusTest);
  statusTestCombo->addStatusTest(dualBlockStatusTest);

  RCP<ParameterList> belosParams = rcp(new ParameterList);
  belosParams->set("Flexible Gmres", false);
  belosParams->set("Num Blocks", 100);
  belosParams->set("Convergence Tolerance", belos_scalar(tol));
  belosParams->set("Maximum Iterations", 100);
  belosParams->set("Verbosity", 33);
  belosParams->set("Output Style", 1);
  belosParams->set("Output Frequency", 1);

  typedef Belos::LinearProblem<belos_scalar, MultiVector, OP> BLinProb;
  RCP<OP> belosOp   = rcp(new Belos::XpetraOp<Scalar, LocalOrdinal, GlobalOrdinal, Node>(blockedMatrix));
  RCP<OP> belosPrec = rcp(new Belos::MueLuOp<Scalar, LocalOrdinal, GlobalOrdinal, Node>(H));

  RCP<tpetra_mvector_type> rhsMultiVector      = reader_type::readDenseFile("f_3dof_mm.txt", comm, fullMap);
  RCP<tpetra_mvector_type> solutionMultiVector = rcp(new tpetra_mvector_type(fullMap, 1));

  RCP<MultiVector> rhsXMultiVector      = rcp(new TpetraMultiVector(rhsMultiVector));
  RCP<MultiVector> solutionXMultiVector = rcp(new TpetraMultiVector(solutionMultiVector));

  RCP<BLinProb> blinproblem = rcp(new BLinProb(belosOp, solutionXMultiVector, rhsXMultiVector));

  blinproblem->setRightPrec(belosPrec);
  blinproblem->setProblem();
  RCP<Belos::SolverManager<belos_scalar, MultiVector, OP>> blinsolver =
      rcp(new Belos::PseudoBlockGmresSolMgr<belos_scalar, MultiVector, OP>(blinproblem, belosParams));

  blinsolver->setUserConvStatusTest(statusTestCombo);

  Belos::ReturnType ret = blinsolver->solve();

  if (ret == Belos::Converged)
    return EXIT_SUCCESS;
  else
    return EXIT_FAILURE;
}

//- -- --------------------------------------------------------
#define MUELU_AUTOMATIC_TEST_ETI_NAME main_
#include "MueLu_Test_ETI.hpp"

int main(int argc, char *argv[]) {
  return Automatic_Test_ETI(argc, argv);
}
