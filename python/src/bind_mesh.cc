// Phase 2: TetMesh binding
#include "bindings.h"
#include <filesystem>

void bind_mesh(py::module_& m)
{
  py::class_<TetMesh>(m, "TetMesh")
    .def(py::init([](py::array_t<double> vertices, py::array_t<int> tets,
                     std::optional<py::array_t<double>> velocities,
                     std::optional<py::array_t<double>> initial_positions) {
      auto vbuf = vertices.unchecked<2>();
      auto tbuf = tets.unchecked<2>();

      if (vbuf.shape(1) != 3)
        throw py::value_error("vertices must have shape (N, 3)");
      if (tbuf.shape(1) != 4)
        throw py::value_error("tets must have shape (M, 4)");

      std::vector<Vector<Real, 3>> verts(vbuf.shape(0));
      for (py::ssize_t i = 0; i < vbuf.shape(0); i++)
        verts[i] = Vector<Real, 3>(vbuf(i, 0), vbuf(i, 1), vbuf(i, 2));

      std::vector<Tetrahedron> tetras(tbuf.shape(0));
      for (py::ssize_t i = 0; i < tbuf.shape(0); i++)
        tetras[i] = Tetrahedron(tbuf(i, 0), tbuf(i, 1), tbuf(i, 2), tbuf(i, 3));

      std::vector<Vector<Real, 3>> vels;
      if (velocities.has_value()) {
        auto velbuf = velocities->unchecked<2>();
        if (velbuf.shape(0) != vbuf.shape(0) || velbuf.shape(1) != 3)
          throw py::value_error("velocities must have shape (N, 3), same N as vertices");
        vels.resize(velbuf.shape(0));
        for (py::ssize_t i = 0; i < velbuf.shape(0); i++)
          vels[i] = Vector<Real, 3>(velbuf(i, 0), velbuf(i, 1), velbuf(i, 2));
      }

      std::vector<Vector<Real, 3>> initPos;
      if (initial_positions.has_value()) {
        auto pbuf = initial_positions->unchecked<2>();
        if (pbuf.shape(0) != vbuf.shape(0) || pbuf.shape(1) != 3)
          throw py::value_error("initial_positions must have shape (N, 3), same N as vertices");
        initPos.resize(pbuf.shape(0));
        for (py::ssize_t i = 0; i < pbuf.shape(0); i++)
          initPos[i] = Vector<Real, 3>(pbuf(i, 0), pbuf(i, 1), pbuf(i, 2));
      }

      return TetMesh(verts, tetras, vels, initPos);
    }), py::arg("vertices"), py::arg("tets"),
       py::arg("velocities") = py::none(),
       py::arg("initial_positions") = py::none(),
       "Construct TetMesh from numpy arrays.\n"
       "  vertices         : (N, 3) rest-shape vertex positions (defines X)\n"
       "  tets             : (M, 4) tetrahedral element indices\n"
       "  velocities       : optional (N, 3) initial velocities\n"
       "  initial_positions: optional (N, 3) initial deformed positions at t=0.\n"
       "                     If given, the body starts deformed (x ≠ X, PE > 0).\n"
       "                     If omitted, x = X (body starts at rest).")

    .def_static("from_file", [](const std::string& path) {
      auto mesh = readTetMeshFromTOBJ(std::filesystem::path(path));
      if (!mesh.has_value())
        throw py::value_error("Failed to load mesh from: " + path);
      return std::move(mesh.value());
    }, py::arg("path"),
       "Load TetMesh from a .tobj file")

    .def_property_readonly("num_vertices", [](const TetMesh& self) {
      return self.getVertices().size();
    }, "Number of vertices in the mesh")

    .def_property_readonly("vertices", [](const TetMesh& self) {
      const auto& verts = self.getVertices();
      auto result = py::array_t<double>({(py::ssize_t)verts.size(), (py::ssize_t)3});
      auto buf = result.mutable_unchecked<2>();
      for (py::ssize_t i = 0; i < (py::ssize_t)verts.size(); i++) {
        buf(i, 0) = verts[i].x();
        buf(i, 1) = verts[i].y();
        buf(i, 2) = verts[i].z();
      }
      return result;
    }, "Vertex positions as numpy array (N, 3). Only available before simulation starts.")

    .def_property_readonly("num_elements", [](const TetMesh& self) {
      return self.tets.size();
    }, "Number of tetrahedral elements")

    .def_property_readonly("elements", [](const TetMesh& self) {
      auto result = py::array_t<int>({(py::ssize_t)self.tets.size(), (py::ssize_t)4});
      auto buf = result.mutable_unchecked<2>();
      for (py::ssize_t i = 0; i < (py::ssize_t)self.tets.size(); i++) {
        buf(i, 0) = self.tets[i][0];
        buf(i, 1) = self.tets[i][1];
        buf(i, 2) = self.tets[i][2];
        buf(i, 3) = self.tets[i][3];
      }
      return result;
    }, "Tet indices as numpy array (M, 4)")

    .def_property_readonly("num_surface_triangles", [](const TetMesh& self) {
      return self.surfaces.size();
    }, "Number of surface triangles");
}
