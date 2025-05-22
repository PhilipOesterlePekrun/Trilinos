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


#include "/home/oesterle/rd/Trilinos_Base/Trilinos/packages/muelu/src/Misc/MueLu_InterfaceAggregationFactory_def.hpp"//#debug

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
  using ST = Teuchos::ScalarTraits<int>;
  using device_type = typename Node::device_type;
  using LO_view = Kokkos::View<LocalOrdinal *, device_type>;
  
  out << "version: " << MueLu::Version() << std::endl;
  
  Level level;
  // Set level ID so 0 so that we can build with our provided mapping
  level.SetLevelID(0);
  
  auto lib = TestHelpers::Parameters::getLib();
  RCP<const Teuchos::Comm<int> > comm = Parameters::getDefaultComm();
  int rank = comm->getRank();
  int commSize = comm->getSize();
  
  Teuchos::ParameterList matrixList;
  // Geometry
  // We choose random problem dimensions of nx in [5,10] and ny in [10,30]
  GO nx, ny;
  if(rank == 0) {
    nx = 5 + ST::random() % 6;
    ny = 10 + ST::random() % 21;
  }
  Teuchos::broadcast(*comm, 0, &nx);
  Teuchos::broadcast(*comm, 0, &ny);
  matrixList.set("nx", nx);
  matrixList.set("ny", ny);
  
  // Distribution
  matrixList.set("mx", 1);
  matrixList.set("my", commSize);
  matrixList.set("matrixType", "Elasticity2D");
  
  // ndofn = number of dofs (degrees of freedom) per node
  constexpr GO ndofnPrimal = 2;
  // nnode = number of nodes
  const GO nnodePrimal = nx*ny;
  LO mynnodePrimal; // maybe can be const, same for below //#
  const GO ndofPrimal = ndofnPrimal*nnodePrimal;
  LO myndofPrimal;
  
  // We choose a random ndofn in [1,3] and a random nnode in [15,25] for the dual (interface) field
  GO ndofnDual;
  GO nnodeDual;
  if(rank == 0) {
    ndofnDual = 1 + ST::random() % 3;
    nnodeDual = 15 + ST::random() % 11;
  }
  Teuchos::broadcast(*comm, 0, &ndofnDual);
  Teuchos::broadcast(*comm, 0, &nnodeDual);
  
  LO mynnodeDual;
  const GO ndofDual = ndofnDual*nnodeDual;
  LO myndofDual;
  
  
  
  
  
  
  // Supply a random, injective global dual to primal mapping
  RCP<Array<GO>> dual2Primal = rcp(new Teuchos::Array<GO>(nnodeDual));
  if(rank == 0) {
    Array<GO> primalCandidates = {};
    for (GO i = 0; i < nnodePrimal; ++i)
      primalCandidates.push_back(i);
    for (GO dualNodeGID = 0; dualNodeGID < nnodeDual; ++dualNodeGID)
      dual2Primal->at(dualNodeGID) = primalCandidates.at(ST::random() % nnodePrimal - 1 - dualNodeGID);
  }
  Teuchos::broadcast(*comm, 0, nnodeDual, dual2Primal->getRawPtr());
  
  // // primal
  // Primal row map is uniform contiguous; nothing fancy
  RCP<const Map> nodeRowMap1 = Xpetra::MapFactory<LO, GO, NO>::createUniformContigMapWithNode(lib, nnodePrimal, comm);
  
  // With global indexing, but this is implied by being GO //#(do this everywhere for consistency)
  ArrayView<const GO> myPrimalNodes = nodeRowMap1->getLocalElementList();
  ///myPrimalNodes.describe(out);//# use better printing or write to files by proc
  mynnodePrimal = nodeRowMap1->getLocalNumElements();//##delete if not needed
  
  //# technically the dual node map creation could go here

  ///RCP<const Map> dofRowMap1 = MueLu::Utils2::helper_buildStridedDofMapFromNodeMap(nodeRowMap1, ndofnPrimal);
  
  RCP<const StridedMap> dofRowMap1 = 
  Teuchos::rcp_dynamic_cast<const StridedMap>(
    MueLu::Utils2::helper_buildStridedDofMapFromNodeMap(nodeRowMap1, ndofnPrimal), true);
    
  ArrayView<const GO> myPrimalDofs = dofRowMap1->getLocalElementList();
  //#dofRowMap1->describe(*outfs, Teuchos::VERB_EXTREME);//#
  // // myDual2Primal and dual map
  Array<GO> myDualNodes;
  
  RCP<std::map<LO, LO>> myDual2Primal = rcp(new std::map<LO, LO>());
  mynnodeDual = 0;
  //# it = iterator. *it is the ref to iterator, ie iterator&. then, to get the std pair from iterator,
  for (GO dualNodeGID = 0; dualNodeGID < nnodeDual; ++dualNodeGID) {
    GO primalGID = dual2Primal->at(dualNodeGID);
    if (nodeRowMap1->isNodeGlobalElement(primalGID)) {
      (*myDual2Primal)[mynnodeDual] = nodeRowMap1->getLocalElement(primalGID);
      myDualNodes.push_back(dualNodeGID);
      ++mynnodeDual;
    }
  }
  
  RCP<const Map> nodeRowMap2 = Xpetra::MapFactory<LO, GO, NO>::Build(
    lib, nnodeDual, myDualNodes, 0, comm);

  RCP<const Map> dofRowMap2 = MueLu::Utils2::helper_buildStridedDofMapFromNodeMap(nodeRowMap2, ndofnDual);
  ArrayView<const GO> myDualDofs = dofRowMap2->getLocalElementList();
  myndofDual = mynnodeDual*ndofnDual;//##delete if not needed

  RCP<Galeri::Xpetra::Problem<Map, CrsMatrixWrap, MultiVector> > Pr =
        Galeri::Xpetra::BuildProblem<SC, LO, GO, Map, CrsMatrixWrap, MultiVector>(matrixList.get("matrixType", ""), dofRowMap1, matrixList);
  RCP<Matrix> A00 = Pr->BuildMatrix();
  A00->SetFixedBlockSize(ndofnPrimal);
    
  RCP<Matrix> A11 = Teuchos::make_rcp<Xpetra::CrsMatrixWrap<SC, LO, GO, NO>>
    (Xpetra::CrsMatrixFactory<SC, LO, GO, NO>::Build(dofRowMap2, 0));
  A11->fillComplete(dofRowMap2, dofRowMap2);
  A11->SetFixedBlockSize(ndofnPrimal);
  
  // The true null space is not necessary
  LO NS1dim = ndofnPrimal;
  RCP<MultiVector> nullspace1 = MultiVectorFactory::Build(dofRowMap1, NS1dim);
  nullspace1->randomize();
  
  // Primal aggregation
  level.Set("A", A00);
  level.Set("nullspace1", nullspace1);
  
  RCP<AmalgamationFactory> amalgFact = rcp(new AmalgamationFactory());
  RCP<CoalesceDropFactory> dropFact  = rcp(new CoalesceDropFactory());
  dropFact->SetFactory("UnAmalgamationInfo", amalgFact);

  RCP<UncoupledAggregationFactory> uncoupledAggFact = rcp(new UncoupledAggregationFactory());
  uncoupledAggFact->SetFactory("Graph", dropFact);
  uncoupledAggFact->SetFactory("Graph", dropFact);
  uncoupledAggFact->SetFactory("DofsPerNode", dropFact);
  level.Set("DofsPerNode", ndofnPrimal);

  /////uncoupledAggFact->SetMinNodesPerAggregate(7);
  uncoupledAggFact->SetOrdering("natural");

  level.Request("Aggregates", uncoupledAggFact.get());
  uncoupledAggFact->Build(level);

  // access aggregates
  RCP<Aggregates> primalAggs         = level.Get<RCP<Aggregates>>("Aggregates", uncoupledAggFact.get()); //# i would test non-null here but its not really the job of this test to test uncoupledAggFact
  
  // Set params
  RCP<InterfaceAggregationFactory> interfaceAggFact = rcp(new InterfaceAggregationFactory());
  
  interfaceAggFact->SetParameter("Dual/primal mapping strategy", Teuchos::ParameterEntry(std::string("node-based")));
  interfaceAggFact->SetParameter("number of DOFs per dual node", Teuchos::ParameterEntry(Teuchos::as<LO>(ndofnDual)));
  level.Set("DualNodeID2PrimalNodeID", myDual2Primal);
  level.Set("A", A11);
  interfaceAggFact->SetFactory("Aggregates", uncoupledAggFact);
  
  interfaceAggFact->Build(level);

  // Request outputs
  level.Request("Aggregates", interfaceAggFact.get(), interfaceAggFact.get());
  level.Request("UnAmalgamationInfo", interfaceAggFact.get(), interfaceAggFact.get());
  
  RCP<Aggregates> dualAggs = level.Get<RCP<Aggregates>>("Aggregates", interfaceAggFact.get());
  TEST_EQUALITY(dualAggs.is_null(), false);
  
  RCP<Aggregates> dualUnAmalgamationInfo = level.Get<RCP<Aggregates>>("UnAmalgamationInfo", interfaceAggFact.get());
  TEST_EQUALITY(dualUnAmalgamationInfo.is_null(), false);

  // Test that the core idea is upheld, for both the current and the next coarser level //#
  //# btw, uncoupledaggregationfactory inherently has aggregatescrossprocessors==false (ofc this isnt the only type of aggregation factory for primal nodes, but we will just use this here then)
  /*
  primalAggs->PrintAllNodesPerAggregate(*outfs, true);//#
  outfs->flush();
    comm->barrier();//#
    if(comm->getRank()==0) std::cout<<"_________barrier_________\n";//#
    comm->barrier();//#
    std::cout << std::flush;
    std::cout<<"\trank"<<comm->getRank()<<"\n";
    comm->barrier();//#
    std::cout << std::flush;
    comm->barrier();//#
    if(comm->getRank()==0) std::cout<<"stdmaps\n";//#
    comm->barrier();//#
    std::cout << std::flush;
    comm->barrier();//#
    std::string myOutput ="std map on rank"+std::to_string(comm->getRank())+"\n";
    for(auto e:*myDual2Primal){
      myOutput+="\t"+std::to_string(e.first)+","+std::to_string(e.second)+"\n";
    }
    std::cout<<myOutput<<"\n";
    comm->barrier();//#
    std::cout << std::flush;
    comm->barrier();//#
  dualAggs->PrintAllNodesPerAggregate(*outfs, true);//#
  outfs->flush();
  */
 
  std::cout<<"\tA00->GetFixedBlockSize();"<<A00->GetFixedBlockSize()<<std::endl;//#
  std::cout<<"\tA11->GetFixedBlockSize();"<<A11->GetFixedBlockSize()<<std::endl;//#
  
  MueLu::Utils2::writeInterleavedPerRank<LO, GO, NO>(comm, {"primalAggs", "dualAggs"}, {primalAggs, dualAggs});//#
  /*
  ArrayRCP<const LO> primalVertex2AggId_mapped = {};
  
  //#dualVertex2AggId[nodeID] returns aggID
  for(LO dualNodeId = 0; dualNodeId < dualVertex2AggId->size(); ++dualNodeId) {
    LO dualAggId = dualVertex2AggId[dualNodeId];
    LO primalNodeId_mapped = myDual2Primal[dualNodeId];
    primalVertex2AggId_mapped->insert(primalNodeId_mapped, primalVertex2AggID[primalNodeId_mapped]);
  }
  */
  
  
  //////// new attempt:
  // Formal check that the core idea (dual aggregates are the restriction of primal aggregates onto the interface) is upheld
  // We do this by checking that the partial primal aggregates which are mapped back from the dual aggregates are a subset of the actual primal aggregates
  
  ArrayRCP<const LO> primalVertex2AggId = primalAggs->GetVertex2AggId()->getData(0);
  ArrayRCP<const LO> dualVertex2AggId = dualAggs->GetVertex2AggId()->getData(0);
  
  bool aggStructureCheck = true;
  
  LO_view aggPtrPrimal;
  LO_view aggNodesPrimal;
  LO_view unaggregatedPrimal;
  primalAggs->ComputeNodesInAggregate(aggPtrPrimal, aggNodesPrimal, unaggregatedPrimal);
  
  LO_view aggPtrDual;
  LO_view aggNodesDual;
  LO_view unaggregatedDual;
  dualAggs->ComputeNodesInAggregate(aggPtrDual, aggNodesDual, unaggregatedDual);
  
  if(unaggregatedPrimal.size() != 0 || unaggregatedDual.size() != 0)
    aggStructureCheck = false;
  
    //# maybe add another check with mynnodePrimal and mynnodeDual in here somewhere
  const LO numDualAggs = dualAggs->GetNumAggregates();
  for(LO dualAggId = 0; dualAggId < numDualAggs && aggStructureCheck; ++dualAggId) {
    LO primalAggId_mapped = -2; // should not change from first iteration onwards
    for (LO dualNodeId = aggPtrDual[dualAggId]; dualNodeId < aggPtrDual[dualAggId+1]; ++dualNodeId)
    {
      // sanity
      if(myDual2Primal->find(dualNodeId) == myDual2Primal->end()) { //!contains
        aggStructureCheck = false;
        break;
      }
      LO primalNodeId_mapped = myDual2Primal->at(dualNodeId);
      
      if(primalNodeId_mapped >= primalVertex2AggId.size()) {
        aggStructureCheck = false;
        break;
      }
      LO primalAggId_mappedTmp = primalVertex2AggId[primalNodeId_mapped];
      
      // check that all of the corresponding primal nodes are also in the same aggregate
      if(primalAggId_mapped == -2)
        primalAggId_mapped = primalAggId_mappedTmp;
      else if(primalAggId_mappedTmp != primalAggId_mapped) {
        aggStructureCheck = false;
        break;
      }
    }
  }
  
  // i think sizes already ensured by above
  
  // Also check all sizes
  //for all procs, for all aggs: primalaggsmapped.size() == dualaggs.size() == myDual2Primal.size()
  bool aggAndMapSizesCheck = true;
  //...
  
  TEST_EQUALITY(true, true);
}

#define MUELU_ETI_GROUP(SC, LO, GO, Node)                                                      \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(InterfaceAggregationFactory, Constructor, SC, LO, GO, Node) \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(InterfaceAggregationFactory, BuildBasedOnNodeMapping, SC, LO, GO, Node)

#include <MueLu_ETI_4arg.hpp>

}  // namespace MueLuTests
