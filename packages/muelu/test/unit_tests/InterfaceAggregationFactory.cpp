#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_DefaultComm.hpp>

#include <MueLu_TestHelpers.hpp>
#include <MueLu_Version.hpp>

#include <Xpetra_MultiVectorFactory.hpp>
#include <Xpetra_VectorFactory.hpp>
#include <Xpetra_Vector.hpp>

#include <MueLu_UncoupledAggregationFactory.hpp>
#include <MueLu_CoalesceDropFactory.hpp>
#include <MueLu_AmalgamationFactory.hpp>
#include <MueLu_CoarseMapFactory.hpp>
#include <MueLu_Aggregates.hpp>
#include <MueLu_BlockedCoarseMapFactory.hpp>
#include <MueLu_InterfaceAggregationFactory.hpp>
#include <MueLu_NoFactory.hpp>

#include <Xpetra_ExportFactory.hpp>//#

#include <Xpetra_MapFactory.hpp>
#include <Xpetra_StridedMapFactory.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_Comm.hpp>


namespace MueLuTests {


template<class LocalOrdinal, class GlobalOrdinal, class Node>
Teuchos::RCP<const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node>>
helper_buildStridedDofMapFromNodeMap(
    const Teuchos::RCP<const Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node>>& nodeMap,
    int dofsPerNode)
{
  using map_type = Xpetra::Map<LocalOrdinal, GlobalOrdinal, Node>;
  using GO = GlobalOrdinal;
  using LO = LocalOrdinal;
  using NO = Node;
  using Teuchos::RCP;
  using Teuchos::Array;
  using Teuchos::ArrayView;

  RCP<const Teuchos::Comm<int>> comm = nodeMap->getComm();
  const GO indexBase = 0;

  // Get local node GIDs owned by this processor
  Teuchos::ArrayView<const GO> nodeGIDs = nodeMap->getLocalElementList();

  std::vector<GO> dofGIDs;
  dofGIDs.reserve(nodeGIDs.size() * dofsPerNode);
  for (GO nodeID : nodeGIDs) {
    for (int d = 0; d < dofsPerNode; ++d) {
      dofGIDs.push_back(nodeID * dofsPerNode + d);
    }
  }

  std::vector<size_t> stridingInfo(1, static_cast<size_t>(dofsPerNode));  // Correct type
  size_t numGlobalDofs = Teuchos::OrdinalTraits<Xpetra::global_size_t>::invalid();
  int stridedBlockId = 0;
  GO offset = 0;

  auto dofMap = Xpetra::StridedMapFactory<LO, GO, Node>::Build(
    nodeMap->lib(),
    numGlobalDofs,
    Teuchos::arrayViewFromVector(dofGIDs),
    indexBase,
    stridingInfo,
    nodeMap->getComm(),
    stridedBlockId,
    offset
  );



  return dofMap;
}

  

TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(InterfaceAggregationFactory, Constructor, Scalar, LocalOrdinal, GlobalOrdinal, Node) {
#include <MueLu_UseShortNames.hpp>
  MUELU_TESTING_SET_OSTREAM;
  MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);
  out << "version: " << MueLu::Version() << std::endl;

  RCP<InterfaceAggregationFactory> interfaceAggFact = rcp(new InterfaceAggregationFactory());
  TEST_EQUALITY(interfaceAggFact != Teuchos::null, true);
}

//
// 2) Test BuildBasedOnNodeMapping end-to-end on a simple 1D Poisson problem
//
TEUCHOS_UNIT_TEST_TEMPLATE_4_DECL(InterfaceAggregationFactory, BuildBasedOnNodeMapping, Scalar, LocalOrdinal, GlobalOrdinal, Node) {
#include "MueLu_UseShortNames.hpp"
  MUELU_TESTING_SET_OSTREAM;
  MUELU_TESTING_LIMIT_SCOPE(Scalar, GlobalOrdinal, Node);
  out << "version: " << MueLu::Version() << std::endl;
  
  //# Xpetra::UseTpetra or Xpetra::UseEpetra
  auto lib = TestHelpers::Parameters::getLib();
  ///auto lib = Xpetra::UseTpetra;
  
  RCP<const Teuchos::Comm<int> > comm = Parameters::getDefaultComm();
  
  /*
  Level level, coarseLevel;
  TestHelpers::TestFactory<Scalar, LO, GO, NO>::createTwoLevelHierarchy(level, coarseLevel);
  RCP<const Teuchos::Comm<int> > comm = Parameters::getDefaultComm();

  GO nx = 2, ny = 3, nz = 4, mx = comm->getSize(), my = 1, mz = 1;
  ;
  LO blkSize = 3;
  Teuchos::ParameterList matrixList;
  matrixList.set("nx", nx);
  matrixList.set("ny", ny);
  matrixList.set("nz", nz);
  matrixList.set("mx", mx);
  matrixList.set("my", my);
  matrixList.set("mz", mz);
  matrixList.set("matrixType", "Elasticity3D");
  RCP<Matrix> Op = TestHelpers::TestFactory<Scalar, LO, GO, NO>::BuildMatrix(matrixList, TestHelpers::Parameters::getLib());
  Op->SetFixedBlockSize(blkSize);
  */

  //level.Set("A", Op);

  
  
  //idea: give the rowmap of A11 explicitly. Also give the range map of A12 explicitly because this is directly used in the BuildBasedOnNodeMapping function (you can prescribe the range map, no problem there)
  
  //RCP<Matrix> A = TestHelpers::TestFactory<SC, LO, GO, NO>::Build1DPoisson(nnodePrimal*ndofnPrimal);
  //A->SetFixedBlockSize(ndofnPrimal);
  //level.Set("A", A);
  
  Level level;
  // Set level ID so 0 so that we can build with our provided mapping
  level.SetLevelID(0);
  
  constexpr GO nx = 5, ny = 10,
    ndofnPrimal = 2, ndofnDual = 2,
    indexBase = 0;
  constexpr GO nnodeDual = 10;
  const GO nnodePrimal = nx*ny;
  
  const GO ndofPrimal = ndofnPrimal*nnodePrimal;
  const GO ndofDual = ndofnDual*nnodeDual;
  
  
  
  /////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////
  //# here i take a lot from framework test to make the contigdof map respecting node ownership and also make the striding maps and all that
  
  std::map<GO, GO> dual2Primal;
  std::map<LO, LO> myDual2Primal;
  read_dual2Primal<GO>(dual2PrimalFileName, dual2Primal);

  // Construct the necessary maps to construct the blocked map
  RCP<const tpetra_map_type> primalNodeMap = Tpetra::createUniformContigMapWithNode<LocalOrdinal, GlobalOrdinal, Node>(numGlobalNodesPrimal, comm);
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

  for (auto i = dual2Primal.begin(); i != dual2Primal.end(); ++i)
    if (primalMap->isNodeGlobalElement(numPrimalDofsPerNode * (i->second)))
      ++numMyDualDofs;

  numMyDualDofs *= numDualDofsPerNode;

  const size_t numMyDofs = numMyPrimalDofs + numMyDualDofs;

  Array<GO> myDualDofs(numMyDualDofs);
  Array<GO> myDofs(numMyDofs);

  for (size_t i = 0; i < numMyPrimalDofs; ++i)
    myDofs[i] = myPrimalDofs[i];

  /* Choose the elements of the primal map, dual map, and of the dual to primal node mapping (myDual2Primal)
   *
   * - The ownership of the primal and dual maps must be chosen such that any pair of dual and primal
   *   indices from the dual2Primal mapping are owned by the same process
   * - The myDual2Primal mapping is then also distributed in the same way
   * - Hence, the initial primalMap decides the distribution of the other data (maps, mapping)
   */
  current_i = 0;
  for (auto i = dual2Primal.begin(); i != dual2Primal.end(); ++i)
    if (primalMap->isNodeGlobalElement(numPrimalDofsPerNode * (i->second))) {
      for (size_t j = 0; j < numDualDofsPerNode; ++j) {
        myDualDofs[numDualDofsPerNode * current_i + j]               = numGlobalDofsPrimal + (i->first) * numDualDofsPerNode + j;
        myDofs[numMyPrimalDofs + numDualDofsPerNode * current_i + j] = numGlobalDofsPrimal + (i->first) * numDualDofsPerNode + j;
      }
      GO primalDof          = numPrimalDofsPerNode * (i->second);
      myDual2Primal[current_i] = primalMap->getLocalElement(primalDof) / numPrimalDofsPerNode;
      ++current_i;
    }

  RCP<const tpetra_map_type> dualMap = rcp(new tpetra_map_type(numGlobalDofsDual, myDualDofs, indexBase, comm));
  RCP<const tpetra_map_type> fullMap = rcp(new tpetra_map_type(numGlobalDofsTotal, myDofs, indexBase, comm));

  RCP<const Map> fullXMap   = rcp(new TpetraMap(fullMap));
  RCP<const Map> primalXMap = rcp(new TpetraMap(primalMap));
  RCP<const Map> dualXMap   = rcp(new TpetraMap(dualMap));

  // Transform the primal and dual maps into strided maps
  std::vector<size_t> stridingInfoPrimal;
  stridingInfoPrimal.push_back(numPrimalDofsPerNode);
  RCP<const StridedMap> stidedPrimalXMap = StridedMapFactory::Build(primalXMap, stridingInfoPrimal);

  std::vector<size_t> stridingInfoDual;
  stridingInfoDual.push_back(numDualDofsPerNode);
  RCP<const StridedMap> stridedDualXMap = StridedMapFactory::Build(dualXMap, stridingInfoDual);

  std::vector<RCP<const Map>> xsubmaps = {stidedPrimalXMap, stridedDualXMap};

  // Construct the blocked map with Xpetra-style indexing
  RCP<const BlockedMap> blockedMap = rcp(new BlockedMap(fullXMap, xsubmaps, false));

  // Read the matrix from file and transform it into a block matrix
  RCP<Matrix> mat = Xpetra::IO<SC, LO, GO, NO>::Read(matrixFileName, fullXMap, fullXMap);
  RCP<MapExtractor> rangeMapExtractor =
      Xpetra::MapExtractorFactory<SC, LO, GO, NO>::Build(fullXMap, xsubmaps);
  RCP<BlockedCrsMatrix> blockedMatrix =
      Xpetra::MatrixUtils<SC, LO, GO, NO>::SplitMatrix(*mat, rangeMapExtractor, rangeMapExtractor);
  blockedMatrix->fillComplete();
  
  
  
  
  
  
  
/////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////
  
  
  
  
  
  
  
  
  
  
  Teuchos::ParameterList matrixList;
  matrixList.set("nx", nx);
  matrixList.set("ny", ny);
  matrixList.set("matrixType", "Elasticity2D");
  // Preliminary because TestFactory doesn't give striding or respect node ownership
  RCP<Matrix> A00prelim = TestHelpers::TestFactory<SC, LO, GO, NO>::BuildMatrix(matrixList, lib); // this function constructs a row map and calls fillComplete() already; however, the row map is not strided...I don't know why honestly, but ok we change that here
  
  /*RCP<const Map> flatMap = A->getRowMap();
  std::vector<size_t> stridingInfo = {2}; // 2 DOFs per node
  RCP<const StridedMap> stridedMap = StridedMapFactory::Build(flatMap, stridingInfo);*/
  
  // dummy row map of length nnodePrimal
  // -> row map of A01
  RCP<const Map> rowNodeMap1 = Xpetra::MapFactory<LO, GO, NO>::Build(lib, nnodePrimal, 0, comm);
  // convert to strided dof row map respecting node ownership
  RCP<const Map> rowDofMap1 = helper_buildStridedDofMapFromNodeMap<LO, GO, NO>(rowNodeMap1, ndofnPrimal);
  
  rowNodeMap1->describe(out);
  rowDofMap1->describe(out);
  
  RCP<Matrix> A00 = Teuchos::make_rcp<Xpetra::CrsMatrixWrap<SC, LO, GO, NO>>(
    Xpetra::CrsMatrixFactory<SC, LO, GO, NO>::Build(rowDofMap1, 0));
  auto exporter = Xpetra::ExportFactory<LO, GO, NO>::Build(A00prelim->getRowMap(), rowDofMap1);
  A00->doExport(*A00prelim, *exporter, Xpetra::INSERT);
  A00->fillComplete();//#rowDofMap1, rowDofMap1); //# actually uses rowmap as rangemap, colmap as domainmap by default with empty args
  ////std::cout<<"\t"<<"typeid(exporter) = "<<typeid(exporter)<<"\n";


  
  // dummy row map of length nnodeDual
  // -> row map of A11 and A10
  RCP<const Map> rowNodeMap2 = Xpetra::MapFactory<LO, GO, NO>::Build(lib, nnodeDual, 0, comm);
  // convert to strided dof row map respecting node ownership
  RCP<const Map> rowDofMap2 = helper_buildStridedDofMapFromNodeMap<LO, GO, NO>(rowNodeMap2, ndofnDual);
  
  // //# ACTUALLY I CAN REPLACE THE LOGIC OF helper_buildStridedDofMapFromNodeMap WITH Xpetra::MapFactory::createUniformContigMapWithNode() AND THEN I ADD THE STRIDING AFTER. JUST LIKE I DID IN THE FRAMEWORK TEST FFS
  
  
  
  RCP<Matrix> A01 = Teuchos::make_rcp<Xpetra::CrsMatrixWrap<SC, LO, GO, NO>>(
    Xpetra::CrsMatrixFactory<SC, LO, GO, NO>::Build(rowDofMap1, 0)); //# CrsMatrix doesnt inherit from Matrix, but BlockedCrsMatrix does??????
  //# ok fine, I will have to build basic matrices, then bring them together into a blockedcrsmatrix, and then use the subblock factories to get the matrices back or something idk

  // dummy domain map of length ndofDual //# just use rowDofMap2
  ///RCP<Map> domainMap01 = Xpetra::MapFactory<LO, GO, NO>::Build(lib, ndofDual, 0, comm);
  // Range map should be shared in blocked matrix
  A01->fillComplete(rowDofMap2, A00->getRangeMap());
  
  //# MAYBE WILL COME IN USEFUL TO USE THIS INSTEAD:
  /*RCP<SubBlockAFactory> A11Fact = Teuchos::rcp(new SubBlockAFactory());
  A11Fact->SetFactory("A", MueLu::NoFactory::getRCP());
  A11Fact->SetParameter("block row", Teuchos::ParameterEntry(0));
  A11Fact->SetParameter("block col", Teuchos::ParameterEntry(0));*/
  
  A00->SetFixedBlockSize(2); // is this necessary?
  level.Set("A", A00);

  //# The true null space of the matrix is not necessary if we already have striding?? ??
  LO NSdim1                   = 2;
  RCP<MultiVector> nullSpace1 = MultiVectorFactory::Build(A00->getRowMap(), NSdim1);
  nullSpace1->randomize();
  level.Set("Nullspace", nullSpace1);

  RCP<AmalgamationFactory> amalgFact = rcp(new AmalgamationFactory());
  RCP<CoalesceDropFactory> dropFact  = rcp(new CoalesceDropFactory());
  dropFact->SetFactory("UnAmalgamationInfo", amalgFact);

  RCP<UncoupledAggregationFactory> UncoupledAggFact = rcp(new UncoupledAggregationFactory());
  UncoupledAggFact->SetFactory("Graph", dropFact);
  UncoupledAggFact->SetFactory("DofsPerNode", dropFact);

  UncoupledAggFact->SetMinNodesPerAggregate(3);
  UncoupledAggFact->SetMaxNeighAlreadySelected(0);
  UncoupledAggFact->SetOrdering("natural");

  RCP<CoarseMapFactory> coarseMapFact = rcp(new CoarseMapFactory());
  coarseMapFact->SetFactory("Aggregates", UncoupledAggFact);

  RCP<BlockedCoarseMapFactory> blockedCoarseMapFact = rcp(new BlockedCoarseMapFactory());
  blockedCoarseMapFact->SetFactory("Aggregates", UncoupledAggFact);
  blockedCoarseMapFact->SetFactory("CoarseMap", coarseMapFact);

  // request input for BlockedCoarseMapFactory by hand
  level.Request("Aggregates", UncoupledAggFact.get());
  level.Request("CoarseMap", coarseMapFact.get());
  level.Request("CoarseMap", blockedCoarseMapFact.get());
  blockedCoarseMapFact->Build(level);
  RCP<const Map> map1 = level.Get<RCP<const Map>>("CoarseMap", coarseMapFact.get());
  RCP<const Map> map2 = level.Get<RCP<const Map>>("CoarseMap", blockedCoarseMapFact.get());

  // access aggregates
  RCP<Aggregates> primalAggs         = level.Get<RCP<Aggregates>>("Aggregates", UncoupledAggFact.get());
  GO numAggs                         = primalAggs->GetNumAggregates();

  // Supply the dual node to primal node mapping
  // Here we use a partial identity mapping (assume ndofDual < ndofPrimal)
  using Dual2Primal = std::map<LO, LO>;
  RCP<Dual2Primal> dual2primal = rcp(new Dual2Primal());
  for (LO i = 0; i < ndofDual; ++i) {
    (*dual2primal)[i] = i;
  }
  
  // Set params
  RCP<InterfaceAggregationFactory> interfaceAggFact = rcp(new InterfaceAggregationFactory());
  //RCP<const Teuchos::ParameterList> interfaceAggFactParamList = interfaceAggFact->GetValidParameterList();
  //auto nb = Teuchos::ParameterEntry();
  //nb.setValue<std::string>()
  interfaceAggFact->SetParameter("Dual/primal mapping strategy", Teuchos::ParameterEntry(std::string("node-based")));
  //interfaceAggFact->SetParameter("DualNodeID2PrimalNodeID", Teuchos::ParameterEntry(dual2primal.get())); //# SetParameter is the same thing as SetFactory lol
  level.Set("DualNodeID2PrimalNodeID", dual2primal);
  interfaceAggFact->SetParameter("number of DOFs per dual node", Teuchos::ParameterEntry(Teuchos::as<LO>(2))); //#this is fine
  ////interfaceAggFact->SetParameter("A", Teuchos::rcp_dynamic_cast<Xpetra::Matrix<SC, LO, GO, NO>>(A01)); //#this is not; even if using Parameter, it stores by value and you dont input matrices like that
  level.Set("A", Teuchos::rcp_dynamic_cast<Xpetra::Matrix<SC, LO, GO, NO>>(A01));
  //level.Set("Dual/primal mapping strategy", "node-based");//"node-based");
  //level.Set("DualNodeID2PrimalNodeID", dual2primal);
  //level.Set("number of DOFs per dual node", Teuchos::as<LO>(2));
  //level.Set("A", A01);

  //interfaceAggFact->SetFactory("A",                       NoFactory::get());
  interfaceAggFact->SetFactory("Aggregates",              UncoupledAggFact);
  //interfaceAggFact->SetFactory("DualNodeID2PrimalNodeID", NoFactory::get());
  
  interfaceAggFact->Build(level);

  // Request outputs
  level.Request("Aggregates", interfaceAggFact.get(), interfaceAggFact.get());
  level.Request("UnAmalgamationInfo", interfaceAggFact.get(), interfaceAggFact.get());
  
  RCP<Aggregates> dualAggs = level.Get<RCP<Aggregates>>("Aggregates", interfaceAggFact.get());

  // Test that the core idea is upheld, for both the current and the next coarser level //#
  //# btw, uncoupledaggregationfactory inherently has aggregatescrossprocessors==false (ofc this isnt the only type of aggregation factory for primal nodes, but we will just use this here then)
  
  primalAggs->PrintAllNodesPerAggregate(out);//#
  dualAggs->PrintAllNodesPerAggregate(out);//#
  TEST_EQUALITY(false, true);
}

#define MUELU_ETI_GROUP(SC, LO, GO, Node)                                                      \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(InterfaceAggregationFactory, Constructor, SC, LO, GO, Node) \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(InterfaceAggregationFactory, BuildBasedOnNodeMapping, SC, LO, GO, Node)

#include <MueLu_ETI_4arg.hpp>

}  // namespace MueLuTests
