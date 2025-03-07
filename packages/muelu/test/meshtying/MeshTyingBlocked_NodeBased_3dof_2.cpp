// @HEADER
// *****************************************************************************
//        MueLu: A package for multigrid based preconditioning
//
// Copyright 2012 NTESS and the MueLu contributors.
// SPDX-License-Identifier: BSD-3-Clause
// *****************************************************************************
// @HEADER

// Belos
#include <BelosPseudoBlockGmresSolMgr.hpp>
#include <BelosStatusTestCombo.hpp>
#include <BelosXpetraStatusTestGenResSubNorm.hpp>
#include <BelosXpetraAdapter.hpp>
#include <BelosMueLuAdapter.hpp>

// MueLu
#include <MueLu_CreateXpetraPreconditioner.hpp>
#include <MueLu_ConfigDefs.hpp>
#include <MueLu_ParameterListInterpreter.hpp>
#include <MueLu_VerboseObject.hpp>
#include "MueLu_AmalgamationFactory.hpp"//#
///#include "MueLu_AmalgamationInfo.hpp"//#

// Teuchos
#include <Teuchos_XMLParameterListHelpers.hpp>

// Xpetra
#include <Xpetra_BlockedMultiVector.hpp>
#include <Xpetra_IO.hpp>
#include <Xpetra_MapExtractor.hpp>
#include <Xpetra_MapExtractorFactory.hpp>
#include <Xpetra_MapUtils.hpp>
#include <Xpetra_MatrixUtils.hpp>

#include <mpi.h>

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

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
int main_(Teuchos::CommandLineProcessor &clp, Xpetra::UnderlyingLib &lib, int argc, char *argv[]) {
#include <MueLu_UseShortNames.hpp>

  // The MMortarSurfaceCoupline_DofBased tests only work with real Scalar types,
  if (Teuchos::ScalarTraits<Scalar>::isComplex) return EXIT_SUCCESS;

  using Teuchos::Array;
  using Teuchos::ArrayRCP;
  using Teuchos::ParameterList;
  using Teuchos::RCP;
  using Teuchos::rcp;

  using SparseMatrixType    = Tpetra::CrsMatrix<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using tpetra_mvector_type = Tpetra::MultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>;
  using tpetra_map_type     = Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>;
  using namespace Teuchos;

  using ST  = Teuchos::ScalarTraits<Scalar>;
  using GST = Teuchos::ScalarTraits<GO>;

  RCP<const Teuchos::Comm<int>> comm = Teuchos::DefaultComm<int>::getComm();
  RCP<Teuchos::FancyOStream> out     = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
  out->setOutputToRootOnly(0);



  
  

  const GO numGlobalDofsPrimal = 2058;
  const GO numGlobalDofsDual = 147;
  const GO numGlobalDofsTotal = numGlobalDofsPrimal + numGlobalDofsDual;
  const int numPrimalDofsPerNode = 3;
  const int numDualDofsPerNode = 3;
  const GO globalPrimalNumNodes = numGlobalDofsPrimal / numPrimalDofsPerNode;

  const std::string xmlFile                  = "simple_3dof_2.xml";
  const std::string matrixFileName           = "MeshTyingBlocked_NodeBased_3dof_matrix.mm";
  const std::string rhsFileName              = "MeshTyingBlocked_NodeBased_3dof_f.mm";
  const std::string nullspace1FileName       = "MeshTyingBlocked_NodeBased_3dof_nullspace1.mm";

  // Create maps for primal DOFs
  /*std::vector<size_t> stridingInfoPrimal;
  stridingInfoPrimal.push_back(numPrimalDofsPerNode);
  RCP<const StridedMap> dofRowMapPrimal = StridedMapFactory::Build(lib, numGlobalDofsPrimal, Teuchos::ScalarTraits<GO>::zero(), stridingInfoPrimal, comm, -1);

  // Create maps for dual DOFs
  std::vector<size_t> stridingInfoDual;
  stridingInfoDual.push_back(numDualDofsPerNode);
  RCP<const StridedMap> dofRowMapDual = StridedMapFactory::Build(lib, numGlobalDofsDual, Teuchos::ScalarTraits<GO>::zero(), stridingInfoDual, comm, -1, numGlobalDofsPrimal);

  // Construct the blocked map of the global system
  std::vector<RCP<const Map>> rowmaps;
  rowmaps.push_back(dofRowMapPrimal);
  rowmaps.push_back(dofRowMapDual);
  RCP<const Map> fullRowMap        = MapUtils::concatenateMaps(rowmaps);
  RCP<const BlockedMap> blockedMap = rcp(new BlockedMap(fullRowMap, rowmaps));*/

  std::map<GO, GO> lagr2Dof;
  std::map<LO, LO> myLagr2Dof;
  read_Lagr2Dof<GO>("Lagr2Dof_3dof.txt", lagr2Dof);

  // Construct the blocked map in Thyra mode
  RCP<const tpetra_map_type> primalNodeMap = Tpetra::createUniformContigMapWithNode<LocalOrdinal, GlobalOrdinal, Node>(globalPrimalNumNodes, comm);
  const GO indexBase                       = primalNodeMap->getIndexBase();
  ArrayView<const GO> myPrimalNodes        = primalNodeMap->getLocalElementList();

  const size_t numMyPrimalNodes = primalNodeMap->getLocalNumElements();
  const size_t numMyPrimalDofs  = numMyPrimalNodes * numPrimalDofsPerNode;

  Array<GO> myPrimalDofs(numMyPrimalDofs);

  LO current_i = 0;
  for (size_t i = 0; i < numMyPrimalNodes; ++i)
    for (size_t j = 0; j < numPrimalDofsPerNode; ++j)
      myPrimalDofs[current_i++] = myPrimalNodes[i] * numPrimalDofsPerNode + j;

  RCP<const tpetra_map_type> primalMap = rcp(new tpetra_map_type(numGlobalDofsPrimal, myPrimalDofs, indexBase, comm));

  size_t numMyDualDofs = 0;

  for (auto i = lagr2Dof.begin(); i != lagr2Dof.end(); ++i)
    if (primalMap->isNodeGlobalElement(numPrimalDofsPerNode * (i->second)))
      ++numMyDualDofs;

  numMyDualDofs *= numDualDofsPerNode;

  const size_t numMyDofs = numMyPrimalDofs + numMyDualDofs;

  Array<GO> myDualDofs(numMyDualDofs);
  Array<GO> myDofs(numMyDofs);

  for (size_t i = 0; i < numMyPrimalDofs; ++i)
    myDofs[i] = myPrimalDofs[i];

  current_i = 0;
  for (auto i = lagr2Dof.begin(); i != lagr2Dof.end(); ++i)
    if (primalMap->isNodeGlobalElement(numPrimalDofsPerNode * (i->second))) {
      for (size_t j = 0; j < numDualDofsPerNode; ++j) {
        myDualDofs[numDualDofsPerNode * current_i + j]               = (i->first) * numDualDofsPerNode + j + numGlobalDofsPrimal;
        myDofs[numMyPrimalDofs + numDualDofsPerNode * current_i + j] = numGlobalDofsPrimal + (i->first) * numDualDofsPerNode + j;
      }
      GO primalDof          = numPrimalDofsPerNode * (i->second);
      myLagr2Dof[current_i] = primalMap->getLocalElement(primalDof) / numPrimalDofsPerNode;
      ++current_i;
    }

  RCP<const tpetra_map_type> dualMap = rcp(new tpetra_map_type(numGlobalDofsDual, myDualDofs, indexBase, comm));
  

  std::cout<<"nodeBased line 121\n";
  RCP<const tpetra_map_type> fullMap = rcp(new tpetra_map_type(numGlobalDofsTotal, myDofs, indexBase, comm));
  primalMap->describe(*Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout)), Teuchos::VERB_EXTREME);
  dualMap->describe(*Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout)), Teuchos::VERB_EXTREME);
  fullMap->describe(*Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout)), Teuchos::VERB_EXTREME);

  RCP<const Map> fullXMap   = rcp(new TpetraMap(fullMap));
  RCP<const Map> primalXMap = rcp(new TpetraMap(primalMap));
  RCP<const Map> dualXMap   = rcp(new TpetraMap(dualMap));



  std::vector<size_t> stridingInfoPrimal;
  stridingInfoPrimal.push_back(numPrimalDofsPerNode);
  RCP<const StridedMap> dofRowMapPrimal = StridedMapFactory::Build(primalXMap, stridingInfoPrimal);

  std::vector<size_t> stridingInfoDual;
  stridingInfoDual.push_back(numDualDofsPerNode);
  RCP<const StridedMap> dofRowMapDual = StridedMapFactory::Build(dualXMap, stridingInfoDual);

  std::cout<<"\n\ntest line 191, strided maps:\n\n";
  dofRowMapPrimal->describe(*Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout)), Teuchos::VERB_EXTREME);
  dofRowMapDual->describe(*Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout)), Teuchos::VERB_EXTREME);



  std::vector<RCP<const Map>> xsubmaps1 = {primalXMap, dualXMap};
  std::vector<RCP<const Map>> xsubmaps = {dofRowMapPrimal, dofRowMapDual};
  RCP<const BlockedMap> blockedMap           = rcp(new BlockedMap(fullXMap, xsubmaps, false));

  // Read input matrices
  typedef Tpetra::MatrixMarket::Reader<SparseMatrixType> reader_type;





  









  // Read the matrix from file and transform it into a block matrix
  RCP<Matrix> mat = Xpetra::IO<SC, LO, GO, NO>::Read(matrixFileName, fullXMap, fullXMap);
  RCP<MapExtractor> rangeMapExtractor =
      Xpetra::MapExtractorFactory<SC, LO, GO, NO>::Build(fullXMap, xsubmaps);

      Teuchos::RCP<const Map> subMap1 = rangeMapExtractor->getMap(0);
      Teuchos::RCP<const Map> subMap2 = rangeMapExtractor->getMap(1);
      std::cout<<"describe submaps:\n";
      subMap1->describe(*Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout)), Teuchos::VERB_EXTREME);
      subMap2->describe(*Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout)), Teuchos::VERB_EXTREME);
      

  RCP<BlockedCrsMatrix> blockedMatrix =
      Xpetra::MatrixUtils<SC, LO, GO, NO>::SplitMatrix(*mat, rangeMapExtractor, rangeMapExtractor);//#
  blockedMatrix->fillComplete();

  // Read the right-hand side vector from file and transform it into a block vector
  RCP<MultiVector> rhs               = Xpetra::IO<SC, LO, GO, NO>::ReadMultiVector(rhsFileName, fullXMap);
  RCP<BlockedMultiVector> blockedRhs = rcp(new BlockedMultiVector(blockedMap, rhs));

  // Read the nullspace vector of the (0,0)-block from file
  RCP<MultiVector> nullspace1 = Xpetra::IO<SC, LO, GO, NO>::ReadMultiVector(nullspace1FileName, primalXMap);

  // Create the default nullspace vector of the (1,1)-block
  RCP<MultiVector> nullspace2 = MultiVectorFactory::Build(dualXMap, 3, true);
  const int dimNS             = 3;
  for (int dim = 0; dim < dimNS; ++dim) {
    ArrayRCP<Scalar> nsValues = nullspace2->getDataNonConst(dim);
    const int numBlocks       = nsValues.size() / dimNS;
    for (int j = 0; j < numBlocks; ++j)
      nsValues[j * dimNS + dim] = Teuchos::ScalarTraits<Scalar>::one();
  }
/////////
  /*std::map<GO, GO> lagr2Dof;
  std::map<LO, LO> myLagr2Dof;
  read_Lagr2Dof<GO>("Lagr2Dof_3dof.txt", lagr2Dof);


  using tpetra_map_type     = Tpetra::Map<LocalOrdinal, GlobalOrdinal, Node>;
  RCP<const tpetra_map_type> primalNodeMap = Tpetra::createUniformContigMapWithNode<LocalOrdinal, GlobalOrdinal, Node>(globalPrimalNumNodes, comm);
  const GO indexBase                       = primalNodeMap->getIndexBase();
  Teuchos::ArrayView<const GO> myPrimalNodes        = primalNodeMap->getLocalElementList();
  const size_t numMyPrimalNodes = primalNodeMap->getLocalNumElements();
  const size_t numMyPrimalDofs  = numMyPrimalNodes * numPrimalDofsPerNode;
  Array<GO> myPrimalDofs(numMyPrimalDofs);

  
std::vector<GlobalOrdinal> myDualNodes = {};
for (size_t i = 0; i < blockedMatrix->getMatrix(0, 1)->getDomainMap()->getLocalNumElements(); i++){
  ///myDualNodes.push_back((operatorRangeMap->getGlobalElement(i) - indexBase) / numDofsPerDualNode + indexBase);
  GlobalOrdinal gDualDofId  = blockedMatrix->getMatrix(0, 1)->getDomainMap()->getGlobalElement(i);
  //std::cout<<"gDualDofId = "<<gDualDofId<<"\n";
  GlobalOrdinal gDualNodeId = AmalgamationFactory::DOFGid2NodeId(gDualDofId, numDualDofsPerNode, numGlobalDofsPrimal, 0);
  //std::cout<<"gDualNodeId = "<<gDualNodeId<<"\n";
  myDualNodes.push_back(gDualNodeId);
}

// remove all duplicates
myDualNodes.erase(std::unique(myDualNodes.begin(), myDualNodes.end()), myDualNodes.end());
std::cout<<"test line 173\n";
std::cout<<"myDualNodes=dualNodes=\n";
for (const auto& node : myDualNodes) {
  std::cout << node << " ";
}
std::cout << std::endl;




//////

int myRank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank); // Get process rank
  MPI_Comm_size(MPI_COMM_WORLD, &size); // Get total number of processes

std::cout<<"test barrier()\n\n";

MPI_Barrier(MPI_COMM_WORLD);
std::cout<<"1myRank= "<<myRank<<"\n"<<std::flush;

MPI_Barrier(MPI_COMM_WORLD);

std::cout<<"2myRank= "<<myRank<<"\n"<<std::flush;

MPI_Barrier(MPI_COMM_WORLD);

std::cout<<"3myRank= "<<myRank<<"\n"<<std::flush;
if(myRank==0) std::cout<<"====================================================================================================================================================================================================================================================================================================================================================================================================================================\n"<<std::flush;

MPI_Barrier(MPI_COMM_WORLD);

std::cout<<"4myRank= "<<myRank<<"\n"<<std::flush;


MPI_Barrier(MPI_COMM_WORLD);


///////////

LO current_i = 0;

//RCP<const Map> A11RangeMap = blockedMatrix->getMatrix(1, 1)->getRangeMap();

for (const auto& entry : lagr2Dof) {
    GO lagrangeNode = entry.first;  // Lagrange node
    GO primalNode   = entry.second; // Corresponding primal node

    // Convert primal node to DOF index
    GO primalDof = primalNode * numPrimalDofsPerNode;

    // Check if this primal DOF is locally owned
    if((blockedMatrix->getMatrix(0, 0))->getDomainMap()->isNodeGlobalElement(primalDof)){
        // Store in local myLagr2Dof
        myLagr2Dof[lagrangeNode] = primalNode;
        ++current_i;
    }
}
*/
  ///////

  RCP<ParameterList> params = Teuchos::getParametersFromXmlFile(xmlFile);
  ParameterListInterpreter mueLuFactory(*params, comm);

  RCP<Hierarchy> hierarchy = mueLuFactory.CreateHierarchy();
  hierarchy->IsPreconditioner(true);
  hierarchy->SetDefaultVerbLevel(MueLu::Extreme);
  hierarchy->GetLevel(0)->Set("A", Teuchos::rcp_dynamic_cast<Matrix>(blockedMatrix));
  hierarchy->GetLevel(0)->Set("Nullspace1", nullspace1);
  //hierarchy->GetLevel(0)->Set("Primal interface DOF map", primalInterfaceDofMap);
  ParameterList &userDataParams = params->sublist("user data");
  ////userDataParams.set<RCP<std::map<LO, LO>>>("DualNodeID2PrimalNodeID", rcpFromRef(myLagr2Dof));
  hierarchy->GetLevel(0)->Set("DualNodeID2PrimalNodeID",
    Teuchos::rcp_dynamic_cast<std::map<int, int>>(Teuchos::rcpFromRef(myLagr2Dof), true));

  mueLuFactory.SetupHierarchy(*hierarchy);//#

  // Create the preconditioned GMRES solver
  using OP = Belos::OperatorT<MultiVector>;

  using blockStatusTestClass = Belos::StatusTestGenResSubNorm<SC, MultiVector, OP>;
  using StatusTestComboClass = Belos::StatusTestCombo<SC, MultiVector, OP>;

  typename ST::magnitudeType tol  = 1e-5;
  typename ST::magnitudeType bTol = 1e-6;

  RCP<blockStatusTestClass> primalBlockStatusTest = rcp(new blockStatusTestClass(bTol, 0));
  RCP<blockStatusTestClass> dualBlockStatusTest   = rcp(new blockStatusTestClass(bTol, 1));

  RCP<StatusTestComboClass> statusTestCombo = rcp(new StatusTestComboClass(StatusTestComboClass::SEQ));
  statusTestCombo->addStatusTest(primalBlockStatusTest);
  statusTestCombo->addStatusTest(dualBlockStatusTest);

  RCP<ParameterList> belosParams = rcp(new ParameterList);
  belosParams->set("Flexible Gmres", false);
  belosParams->set("Num Blocks", 100);
  belosParams->set("Convergence Tolerance", tol);
  belosParams->set("Maximum Iterations", 100);
  belosParams->set("Verbosity", 33);
  belosParams->set("Output Style", 1);
  belosParams->set("Output Frequency", 1);

  using BLinProb    = Belos::LinearProblem<SC, MultiVector, OP>;
  RCP<OP> belosOp   = rcp(new Belos::XpetraOp<Scalar, LocalOrdinal, GlobalOrdinal, Node>(blockedMatrix));
  RCP<OP> belosPrec = rcp(new Belos::MueLuOp<Scalar, LocalOrdinal, GlobalOrdinal, Node>(hierarchy));

  RCP<MultiVector> X = MultiVectorFactory::Build(fullXMap, 1, true);

  RCP<BLinProb> blinproblem = rcp(new BLinProb(belosOp, X, rhs));

  blinproblem->setRightPrec(belosPrec);
  blinproblem->setProblem();
  RCP<Belos::SolverManager<SC, MultiVector, OP>> blinsolver =
      rcp(new Belos::PseudoBlockGmresSolMgr<SC, MultiVector, OP>(blinproblem, belosParams));

  blinsolver->setUserConvStatusTest(statusTestCombo);

  Belos::ReturnType ret   = blinsolver->solve();
  const int numIterations = blinsolver->getNumIters();

  if (ret == Belos::Converged)
    return EXIT_SUCCESS;
  else
    return EXIT_FAILURE;
}

//-----------------------------------------------------------
#define MUELU_AUTOMATIC_TEST_ETI_NAME main_
#include "MueLu_Test_ETI.hpp"

int main(int argc, char *argv[]) {
  return Automatic_Test_ETI(argc, argv);
}
