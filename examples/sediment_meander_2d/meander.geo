Mesh.CharacteristicLengthFromPoints = 1;

Point (1) = {0, -9, 0, 0.1};
Point (101) = {0.5, -9, 0, 0.25};
Point (2) = {0, 2.5, 0, 0.1};
Point (3) = {1, -9, 0, 0.1};
Point (4) = {1, 2.5, 0, 0.1};
Point (14) = {0.5, 2.5, 0, 0.25};
Point (5) = {8, 2.5, 0, 0.1};
Point (12) = {8.5, 2.5, 0, 0.25};
Point (6) = {9, 2.5, 0, 0.1};
Point (7) = {8, -9, 0, 0.1};
Point (107) = {8.5, -9, 0, 0.25};
Point (8) = {9, -9, 0, 0.1};
Point (9) = {4.5, 7,0, 0.1};
Point (10) = {4.5, 2.5, 0, 0.1};
Point (11) = {4.5, 6, 0, 0.1};
Point (13) = {4.5, 6.5, 0, 0.25};
Circle (9) = {2, 10, 9};
Circle (10) = {4, 10, 11};
Circle (11) = {12, 10, 13};
Circle (12) = {13, 10, 14};
Circle (3) = {9, 10,  6};
Circle (4) = {11, 10, 5};
Line (5) = {1,2};
Line (6) = {3, 4};
Line (7) = {7, 5};
Line (8) = {8, 6};

Line(34) = {101, 14};
Line(35) = {101, 1};
Line(36) = {3, 101};
Line(37) = {12, 107};
Line(38) = {7, 107};
Line(39) = {8, 107};


Physical Line(1) = {35, 36};
Physical Line(2) = {38, 39};
Physical Line(3) = {3, 4, 5, 6, 7, 8 ,9, 10};

Line Loop (20) = {35, 5, 9, 3, -8, 39, -37, 11, 12, -34};
Plane Surface(16) = {20};
Physical Surface(200) = {20};

Line Loop (30) = {-36, 6, 10, 4, -7, 38, -37, 11, 12, -34};
Plane Surface(20) = {30};
Physical Surface(100) = {16};
