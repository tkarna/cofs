Lx = 724000;
Ly = 764000;
Lriver = 45000;
Wriver = 2000;  // for testing only
Yriver = 30000;
lc = 1e6;
L = 5;
Lup = 0;

Point(1) = {0, 0, 0, lc};
Point(2) = {-Lx , 0, 0, lc};
Point(3) = {-Lx , Ly , 0, lc};
Point(4) = {0, Ly , 0, lc};
Point(5) = {0, Yriver+Wriver/2 , 0, lc};
Point(6) = {Lriver, Yriver+Wriver/2 , 0, lc};
Point(7) = {Lriver, Yriver-Wriver/2 , 0, lc};
Point(8) = {0, Yriver-Wriver/2 , 0, lc};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,5};
Line(5) = {5,6};
Line(6) = {6,7};
Line(7) = {7,8};
Line(8) = {8,1};

Physical Line(1) = {1}; // south
Physical Line(2) = {2};  // west
Physical Line(3) = {3}; // north
Physical Line(4) = {4,8};  // coast
Physical Line(5) = {5,7}; // river sides
Physical Line(6) = {6}; // river inlet

Line Loop(5) = {1,2,3,4,5,6,7,8};
Plane Surface(6) = {5};
Physical Surface(7) = {6};

Field[1] = MathEval;
Field[1].F = "(120e3-1.5e3)*((y-80e3)/(764e3-80e3)) + 2e3";
Field[2] = MathEval;
Field[2].F = "(120e3-1.5e3)*((-y+3e3)/(764e3-3e3)) + 2e3";
Field[3] = MathEval;
Field[3].F = "(200e3-1.5e3)*((x+30e3)/(-724e3+30e3)) + 2e3";
Field[4] = Max;
Field[4].FieldsList = {1, 2};
Field[5] = Max;
Field[5].FieldsList = {3, 4};
Field[6] = MathEval;
Field[6].F = "2000";
Field[7] = Max;
Field[7].FieldsList = {5, 6};
Field[8] = Attractor;
Field[8].NodesList = {5, 8};
Field[8].EdgesList = {5, 7};
Field[8].NNodesByEdge = 200;
Field[9] = Threshold;
Field[9].DistMax = 10000;
Field[9].DistMin = 1000;
Field[9].LcMax = 3000;
Field[9].LcMin = 1000;
Field[9].IField = 8;
Field[10] = Max;
Field[10].FieldsList = {7, 9};
Background Field = 10;

Mesh.Algorithm = 6; // frontal=6, delannay=5, meshadapt=1
