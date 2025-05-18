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

namespace MueLuTests {

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
  
  constexpr GO nx = 4, ny = 5,
    ndofnPrimal = 2, ndofnDual = 2,
    indexBase = 0;
  constexpr GO nnodeDual = 5;
  const GO nnodePrimal = nx*ny;
  
  const GO ndofPrimal = ndofnPrimal*nnodePrimal;
  const GO ndofDual = ndofnDual*nnodeDual;
  
  Teuchos::ParameterList matrixList;
  matrixList.set("nx", nx);
  matrixList.set("ny", ny);
  matrixList.set("matrixType", "Elasticity2D");
  RCP<Matrix> A = TestHelpers::TestFactory<SC, LO, GO, NO>::BuildMatrix(matrixList, TestHelpers::Parameters::getLib());
  
  // dummy row map
  RCP<Map> A01rowMap = Xpetra::MapFactory<LO, GO, NO>::Build(Xpetra::UseTpetra,
                                                    ndofPrimal,
                                                    0,
                                                    comm);

  RCP<CrsMatrix> A01 = Xpetra::CrsMatrixFactory<SC, LO, GO, NO>::Build(A01rowMap, 0);

  // Range map should be shared in blocked matrix
  A01->fillComplete(A->getDomainMap(), A->getRangeMap());
  
  //# MAYBE WILL COME IN USEFUL TO USE THIS INSTEAD:
  /*RCP<SubBlockAFactory> A11Fact = Teuchos::rcp(new SubBlockAFactory());
  A11Fact->SetFactory("A", MueLu::NoFactory::getRCP());
  A11Fact->SetParameter("block row", Teuchos::ParameterEntry(0));
  A11Fact->SetParameter("block col", Teuchos::ParameterEntry(0));*/
  
  A->SetFixedBlockSize(2);
  level.Set("A", A);

  // The true null space of the matrix is not necessary
  LO NSdim                   = 2;
  RCP<MultiVector> nullSpace = MultiVectorFactory::Build(A->getRowMap(), NSdim);
  nullSpace->randomize();
  level.Set("Nullspace", nullSpace);

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
  level.Set("DualNodeID2PrimalNodeID", dual2primal.get());
  interfaceAggFact->SetParameter("number of DOFs per dual node", Teuchos::ParameterEntry(Teuchos::as<LO>(2)));
  //interfaceAggFact->SetParameter("A", Teuchos::ParameterEntry(A01));
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
  
  primalAggs->PrintAllNodesPerAggregate(out);//#
  dualAggs->PrintAllNodesPerAggregate(out);//#
  TEST_EQUALITY(false, true);
}

#define MUELU_ETI_GROUP(SC, LO, GO, Node)                                                      \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(InterfaceAggregationFactory, Constructor, SC, LO, GO, Node) \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(InterfaceAggregationFactory, BuildBasedOnNodeMapping, SC, LO, GO, Node)

#include <MueLu_ETI_4arg.hpp>

}  // namespace MueLuTests
