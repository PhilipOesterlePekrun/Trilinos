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
  out << "version: " << MueLu::Version() << std::endl;
  
  RCP<Teuchos::FancyOStream> outfs = Teuchos::fancyOStream(rcpFromRef(std::cout));
  
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
  
  constexpr GO nx = 5, ny = 15;
  
  Teuchos::ParameterList matrixList;
  // geometry
  matrixList.set("nx", nx);
  matrixList.set("ny", ny);
  // procs
  matrixList.set("mx", 1);
  matrixList.set("my", comm->getSize());
  matrixList.set("matrixType", "Elasticity2D");
  
  constexpr GO indexBase = 0,
    // ndofn = number of dofs per node
    ndofnPrimal = 2, ndofnDual = 2;
  // nnode = number of nodes
  // - if not my, then global
  const GO nnodePrimal = nx*ny;
  LO mynnodePrimal; // maybe can be const, same for below //#
  constexpr GO nnodeDual = 20;
  LO mynnodeDual;
  // ndof = number of dofs
  const GO ndofPrimal = ndofnPrimal*nnodePrimal;
  LO myndofPrimal;
  const GO ndofDual = ndofnDual*nnodeDual;
  LO myndofDual;
  
  /////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////
  // //# here i take a lot from framework test to make the contigdof map respecting node ownership and also make the striding maps and all that
  
  using Dual2Primal = std::map<LO, LO>;
  
  RCP<Dual2Primal> dual2Primal = rcp(new Dual2Primal());
  
  // Supply the global dual node to primal node mapping
  // Here we use a partial identity mapping (assume nnodeDual < nnodePrimal)
  for (LO i = 0; i < nnodeDual; ++i) {
    (*dual2Primal)[i] = i;
  }
  
  // // primal
  // Primal row map is uniform contiguous; nothing fancy
  RCP<const Map> nodeRowMap1 = Xpetra::MapFactory<LO, GO, NO>::createUniformContigMapWithNode(lib, nnodePrimal, comm);
  
  // With global indexing, but this is implied by being GO //#(do this everywhere for consistency)
  ArrayView<const GO> myPrimalNodes = nodeRowMap1->getLocalElementList();
  ///myPrimalNodes.describe(out);//# use better printing or write to files by proc
  mynnodePrimal = nodeRowMap1->getLocalNumElements();
  myndofPrimal  = nnodePrimal * ndofnPrimal;
  
  //# technically the dual node map creation could go here

  ///RCP<const Map> dofRowMap1 = MueLu::Utils2::helper_buildStridedDofMapFromNodeMap(nodeRowMap1, ndofnPrimal);
  
  RCP<const StridedMap> dofRowMap1 = 
  Teuchos::rcp_dynamic_cast<const StridedMap>(
    MueLu::Utils2::helper_buildStridedDofMapFromNodeMap(nodeRowMap1, ndofnPrimal), true);
    
  ArrayView<const GO> myPrimalDofs = dofRowMap1->getLocalElementList();
  dofRowMap1->describe(*outfs, Teuchos::VERB_EXTREME);//#
  // // myDual2Primal and dual map
  Array<GO> myDualNodes;
  
  RCP<Dual2Primal> myDual2Primal = rcp(new Dual2Primal());
  mynnodeDual = 0;
  //# it = iterator. *it is the ref to iterator, ie iterator&. then, to get the std pair from iterator,
  for (auto it = dual2Primal->begin(); it != dual2Primal->end(); ++it) {
    GO primalGID = it->second;
    if (nodeRowMap1->isNodeGlobalElement(primalGID)) {
      //myDual2Primal->emplace(*it); //# or &(*it)? //# this would do global to global
      (*myDual2Primal)[mynnodeDual] = nodeRowMap1->getLocalElement(primalGID);
      myDualNodes.push_back(it->first);
      ++mynnodeDual;
    }
  }
  
  RCP<const Map> nodeRowMap2 = Xpetra::MapFactory<LO, GO, NO>::Build(
    lib, nnodeDual, myDualNodes, indexBase, comm);

  RCP<const Map> dofRowMap2 = MueLu::Utils2::helper_buildStridedDofMapFromNodeMap(nodeRowMap2, ndofnDual);
  ArrayView<const GO> myDualDofs = dofRowMap2->getLocalElementList();
  ///myDualDofs->describe(out);//#
  myndofDual = mynnodeDual*ndofnDual;
  
  dofRowMap2->describe(*outfs, Teuchos::VERB_EXTREME);//#

  // // Make matrices
  // Preliminary because TestFactory doesn't give striding or respect node ownership
  ///RCP<Matrix> A00prelim = TestHelpers::TestFactory<SC, LO, GO, NO>::BuildMatrix(matrixList, lib); // this function constructs a row map and calls fillComplete() already; however, the row map is not strided...I don't know why honestly, but ok we change that here
  
  /*
  RCP<Matrix> A00 = Teuchos::make_rcp<Xpetra::CrsMatrixWrap<SC, LO, GO, NO>>
    (Xpetra::CrsMatrixFactory<SC, LO, GO, NO>::Build(dofRowMap1, 0));
  auto exporter = Xpetra::ExportFactory<LO, GO, NO>::Build(A00prelim->getRowMap(), dofRowMap1);
  A00->doExport(*A00prelim, *exporter, Xpetra::INSERT);
  A00->fillComplete();//#dofRowMap1, dofRowMap1);
  */
  //# actually uses rowmap as rangemap, colmap as domainmap by default with empty args
  ////std::cout<<"\t"<<"typeid(exporter) = "<<typeid(exporter)<<"\n";
  //# alt to above:
  RCP<Galeri::Xpetra::Problem<Map, CrsMatrixWrap, MultiVector> > Pr =
        Galeri::Xpetra::BuildProblem<SC, LO, GO, Map, CrsMatrixWrap, MultiVector>(matrixList.get("matrixType", "Laplace1D"), dofRowMap1, matrixList);
    RCP<Matrix> A00 = Pr->BuildMatrix();
    
    ///A00->resumeFill();
    ///A00->fillComplete(dofRowMap1, dofRowMap1);
    ///A00->CreateView("stridedMaps", dofRowMap1, A00->getColMap());
    A00->SetFixedBlockSize(ndofnPrimal);
    
    ////A00->CreateView("stridedMaps",
        /////      dofRowMap1,     // range map
             //// dofRowMap1);     // domain map

  std::cout<<"test line 253:"<<matrixList.get("matrixType", "Laplace1D")<<"\n";
    
  RCP<Matrix> A11 = Teuchos::make_rcp<Xpetra::CrsMatrixWrap<SC, LO, GO, NO>>
    (Xpetra::CrsMatrixFactory<SC, LO, GO, NO>::Build(dofRowMap2, 0));
  // Note: for BuildBasedOnNodeMapping, the only map from A11 that is used is the rangemap
  
  A11->fillComplete(dofRowMap2, dofRowMap2);
  A11->SetFixedBlockSize(ndofnPrimal);
  
  // // Make nullspace1
  //# The true null space of the matrix is not necessary if we already have striding?? ??
  LO NS1dim = 2;
  RCP<MultiVector> nullspace1 = MultiVectorFactory::Build(dofRowMap1, NS1dim);
  nullspace1->randomize();
  
  //#A11->getRangeMap()->describe(out, Teuchos::VERB_EXTREME);//#
  
  // // Assign for uncoupled aggregation
  
  level.Set("Nullspace", nullspace1);
  
  level.Set("A", A00);
  level.Set("nullspace1", nullspace1);
  
  
  
  
/////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////
  /////////////////////////////////////////////////////////
  
  
  
  RCP<AmalgamationFactory> amalgFact = rcp(new AmalgamationFactory());
  RCP<CoalesceDropFactory> dropFact  = rcp(new CoalesceDropFactory());
  dropFact->SetFactory("UnAmalgamationInfo", amalgFact);

  RCP<UncoupledAggregationFactory> UncoupledAggFact = rcp(new UncoupledAggregationFactory());
  UncoupledAggFact->SetFactory("Graph", dropFact);
  UncoupledAggFact->SetFactory("Graph", dropFact);
  UncoupledAggFact->SetFactory("DofsPerNode", dropFact);
  ///UncoupledAggFact->SetParameter("DofsPerNode", Teuchos::ParameterEntry(Teuchos::as<LO>(ndofnPrimal))); //# I cant just set this? What?
  level.Set("DofsPerNode", ndofnPrimal);

  UncoupledAggFact->SetMinNodesPerAggregate(7);
  ///UncoupledAggFact->SetMaxNodesPerAggregate(10);
  ///UncoupledAggFact->SetMaxNeighAlreadySelected(0);
  UncoupledAggFact->SetOrdering("natural");

  // request input for BlockedCoarseMapFactory by hand
  level.Request("Aggregates", UncoupledAggFact.get());
  UncoupledAggFact->Build(level);

  // access aggregates
  RCP<Aggregates> primalAggs         = level.Get<RCP<Aggregates>>("Aggregates", UncoupledAggFact.get());
  GO numAggs                         = primalAggs->GetNumAggregates();

  
  
  // Set params
  RCP<InterfaceAggregationFactory> interfaceAggFact = rcp(new InterfaceAggregationFactory());
  //RCP<const Teuchos::ParameterList> interfaceAggFactParamList = interfaceAggFact->GetValidParameterList();
  //auto nb = Teuchos::ParameterEntry();
  //nb.setValue<std::string>()
  interfaceAggFact->SetParameter("Dual/primal mapping strategy", Teuchos::ParameterEntry(std::string("node-based")));
  //interfaceAggFact->SetParameter("DualNodeID2PrimalNodeID", Teuchos::ParameterEntry(dual2primal.get())); //# SetParameter is the same thing as SetFactory lol
  level.Set("DualNodeID2PrimalNodeID", myDual2Primal);
  interfaceAggFact->SetParameter("number of DOFs per dual node", Teuchos::ParameterEntry(Teuchos::as<LO>(ndofnDual))); //#this is fine
  ////interfaceAggFact->SetParameter("A", Teuchos::rcp_dynamic_cast<Xpetra::Matrix<SC, LO, GO, NO>>(A11)); //#this is not; even if using Parameter, it stores by value and you dont input matrices like that
  level.Set("A", A11);
  ///Teuchos::rcp_dynamic_cast<Xpetra::Matrix<SC, LO, GO, NO>>(A11)
  
  //level.Set("Dual/primal mapping strategy", "node-based");//"node-based");
  //level.Set("DualNodeID2PrimalNodeID", dual2primal);
  //level.Set("number of DOFs per dual node", Teuchos::as<LO>(2));
  //level.Set("A", A11);

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
 
  std::cout<<"\tA00->GetFixedBlockSize();"<<A00->GetFixedBlockSize()<<std::endl;
  std::cout<<"\tA11->GetFixedBlockSize();"<<A11->GetFixedBlockSize()<<std::endl;
  
  MueLu::Utils2::writeInterleavedPerRank<LO, GO, NO>(comm, {"primalAggs", "dualAggs"}, {primalAggs, dualAggs});
  
  TEST_EQUALITY(true, true);
}

#define MUELU_ETI_GROUP(SC, LO, GO, Node)                                                      \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(InterfaceAggregationFactory, Constructor, SC, LO, GO, Node) \
  TEUCHOS_UNIT_TEST_TEMPLATE_4_INSTANT(InterfaceAggregationFactory, BuildBasedOnNodeMapping, SC, LO, GO, Node)

#include <MueLu_ETI_4arg.hpp>

}  // namespace MueLuTests
