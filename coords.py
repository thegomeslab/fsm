import itertools
import numpy as np
import networkx as nx
from ase.data import covalent_radii, vdw_radii
from ase.units import Bohr
from collections import OrderedDict
from scipy.spatial.distance import euclidean
from scipy.interpolate import CubicSpline
from scipy.optimize import lsq_linear
from geom import project_trans_rot
from geometric.internal import Distance, Angle, LinearAngle, Dihedral, OutOfPlane, \
                                CartesianX, CartesianY, CartesianZ

angs_to_bohr = 1/Bohr
deg_to_rad = np.pi/180.

class Coordinates(object):

    def __init__(self, atoms1, atoms2=None, verbose=False):
        self.atoms1 = atoms1
        self.atoms2 = atoms2
        self.coords = self.construct()
        self.keys = list(self.coords.keys())
        self.verbose = verbose
        if self.atoms2 is not None and verbose:
           self.dqprint(self.atoms1, self.atoms2)
        elif verbose:
           self.qprint(self.atoms1)
        #self.dqprint(self.atoms1, self.atoms2)
        

    def construct(self):
        raise NotImplementedError("No construct function")

    def qprint(self, atoms):
        xyz = atoms.get_positions()
        xyzb = xyz * angs_to_bohr
        print("\n%15s%15s" % ('Coordinate', 'Value'))
        for name, coord in self.coords.items():
            print("%15s = %15.8f" % (name, coord.value(xyzb)))

    def q(self, xyz):
        xyzb = xyz * angs_to_bohr
        return np.array([coord.value(xyzb) for coord in self.coords.values()])

    def dqprint(self, atoms1, atoms2):
        q1 = self.q(atoms1.get_positions())
        q2 = self.q(atoms2.get_positions())
        dq = q2-q1
        print("\n%15s%15s" % ('Coordinate', 'Value'))
        for i, (name, coord) in enumerate(self.coords.items()):
            star = ""
            if ("bend" in name or "tors" in name or "oop" in name) and dq[i] < -np.pi:
                star = "*"
            elif ("bend" in name or "tors" in name or "oop" in name) and dq[i] > np.pi:
                star = "*"
            print("%15s = %15.8f %15.8f %15.8f %s" % (name, q1[i], q2[i], dq[i], star))

    def b_matrix(self, xyz):
        xyzb = xyz * angs_to_bohr
        nint = len(self.coords)
        ncart = xyzb.size
        B = np.zeros((nint, ncart))
        for i, coord in enumerate(self.coords.values()):
            B[i] = coord.derivative(xyzb).flatten()
        return B

    def u_matrix(self, Bprim):
        evals, evecs = np.linalg.eigh(Bprim@Bprim.T)
        U = evecs[:, evals>1e-8]
        return U

    def x(self, xyz, qtarget):

        xyz1 = xyz.copy()
        for i, name in enumerate(self.keys):
            if "linearbnd" in name:
              self.coords[name].reset(xyz1*angs_to_bohr)

        q0 = self.q(xyz1)
        dq = qtarget - q0
        for i, name in enumerate(self.keys):
            if ("tors" in name) and dq[i] < -np.pi: dq[i] += 2*np.pi
            elif ("tors" in name) and dq[i] > np.pi: dq[i] -= 2*np.pi
        Bprim = self.b_matrix(xyz1)
        U = self.u_matrix(Bprim)
        B = U.T@Bprim
        BT_inv = np.linalg.pinv(B@B.T)@B
        dq = U.T@dq
        dx = BT_inv.T@dq
        rms_dx = np.sqrt(np.mean(dx**2))
        rms_dq = np.sqrt(np.mean(dq**2))
        xyz_backup = xyz1.copy() + dx.reshape(-1, 3) / angs_to_bohr
        dq_min = rms_dq

        niter = 1
        while rms_dx > 1e-7:

            xyz1 += dx.reshape(-1, 3) / angs_to_bohr

            q0 = self.q(xyz1)
            dq = qtarget-q0
            for i, name in enumerate(self.keys):
                if ("tors" in name) and dq[i] < -np.pi: dq[i] += 2*np.pi
                elif ("tors" in name) and dq[i] > np.pi: dq[i] -= 2*np.pi
            Bprim = self.b_matrix(xyz1)
            U = self.u_matrix(Bprim)
            B = U.T@Bprim
            BT_inv = np.linalg.pinv(B@B.T)@B
            dq = U.T@dq
            dx = BT_inv.T@dq
            rms_dx = np.sqrt(np.mean(dx**2))
            rms_dq = np.sqrt(np.mean(dq**2))

            niter += 1
            if niter > 200:
                if self.verbose: print("R FUNCTION FAILED")
                if self.verbose: print("Iteration %d" % niter)
                if self.verbose: print("\tRMS(dx) = %10.5e" % rms_dx)
                if self.verbose: print("\tRMS(dq) = %10.5e" % rms_dq)
                return xyz_backup

            if rms_dq < dq_min:
                xyz_backup = xyz1.copy()

        return xyz1

class Cartesian(Coordinates):

    def construct(self):

        coords = {}
        natoms = len(self.atoms1.numbers)
        for i in range(natoms):
            coords['cartx_{}'.format(i)] = CartesianX(i, w=1.0)
            coords['carty_{}'.format(i)] = CartesianY(i, w=1.0)
            coords['cartz_{}'.format(i)] = CartesianZ(i, w=1.0)
        return coords

class Redundant(Coordinates):

    def checkstre(self, A, B, eps=1e-08):
        v0 = A-B
        n = np.maximum(1e-12, v0.dot(v0))
        if n < eps: return False
        else: return True

    def checkangle(self, A, B, C):
        if self.checkstre(A, B) and self.checkstre(B, C): return True
        else: return False

    def checktors(self, A, B, C, D):
        check1 = self.checkstre(A, B)
        check2 = self.checkstre(B, C)
        check3 = self.checkstre(C, D)
        if check1 and check2 and check3: return True
        else: return False

    def get_fragments(self, A):
      G = nx.to_networkx_graph(A)
      frags = [np.array(list(d)) for d in nx.connected_components(G)]
      return frags

    def connectivity(self, atoms):

      # this is done in Angstrom
      xyz = atoms.get_positions()
      z = atoms.get_atomic_numbers()
      natoms = len(z)

      # compute covalent bonds
      conn = np.zeros((natoms, natoms), dtype=int)
      for i,j in itertools.combinations(range(natoms), 2):
        R = euclidean(xyz[i], xyz[j])
        Rcov = (covalent_radii[z[i]] + covalent_radii[z[j]])
        if R < 1.3 * Rcov:
          conn[i, j] = conn[j, i] = 1

      # find all fragments
      frags = self.get_fragments(conn)
      nfrags = len(frags)

      conn_frag = np.zeros((natoms, natoms), dtype=int)
      conn_frag_aux = np.zeros((natoms, natoms), dtype=int)
      conn_frag_idx = np.zeros((nfrags, nfrags, 2), dtype=int)
      conn_frag_dist = np.zeros((nfrags, nfrags), dtype=float)

      # if fragments>1 get interfragment bonds
      if nfrags > 1:

        # find closest interfragment distances
        for i,j in itertools.combinations(range(natoms), 2):
          i_frag = np.argmax([i in frag for frag in frags])
          j_frag = np.argmax([j in frag for frag in frags])
          if i_frag != j_frag:
            # check distance
            conn_frag_ij = conn_frag_dist[i_frag, j_frag]
            R = euclidean(xyz[i], xyz[j])
            if conn_frag_ij == 0. or conn_frag_ij > R:
              conn_frag_dist[i_frag, j_frag] = conn_frag_dist[j_frag, i_frag] = R
              conn_frag_idx[i_frag, j_frag] = [i, j]
              conn_frag_idx[j_frag, i_frag] = [j, i]

        # set interfrag connectivity matrix
        for i_frag in range(nfrags):
          for j_frag in range(i_frag+1, nfrags):
            i,j = conn_frag_idx[i_frag, j_frag]
            conn_frag[i, j] = conn_frag[j, i] = 1

        # auxillary interfragment bonds are < 2 Ang or < 1.3*interfrag distance
        for i,j in itertools.combinations(range(natoms), 2):
          i_frag = np.argmax([i in frag for frag in frags])
          j_frag = np.argmax([j in frag for frag in frags])
          if i_frag != j_frag:
            conn_frag_ij = conn_frag_dist[i_frag, j_frag]
            R = euclidean(xyz[i], xyz[j])
            if R<2.0 or R<1.3*conn_frag_ij:
              conn_frag_aux[i, j] = conn_frag_aux[j, i] = 1
        conn_frag_aux = conn_frag_aux - conn_frag

      # find hydrogen bond hydrogens
      X_atnum = [7, 8, 9, 15, 16, 17] # N, O, F, P, S, Cl
      is_hbond_h = np.zeros((natoms,), dtype=int)
      for i,j in itertools.combinations(range(natoms), 2):
        if (z[i] == 1 and z[j] in X_atnum):
          if conn[i, j]:
              is_hbond_h[i] = 1
        elif (z[j] == 1 and z[i] in X_atnum):
          if conn[i, j]:
              is_hbond_h[j] = 1

      # find hydrogen bonds
      conn_hbond = np.zeros((natoms, natoms), dtype=int)
      for i,j in itertools.combinations(range(natoms), 2):
        if is_hbond_h[i] and not conn[i, j] and z[j] in X_atnum:
          R = euclidean(xyz[i], xyz[j])
          Rvdw = vdw_radii[z[i]] + vdw_radii[z[j]]
          if R < 0.9*Rvdw:
            conn_hbond[i, j] = conn_hbond[j, i] = 1
        elif is_hbond_h[j] and not conn[i, j] and z[i] in X_atnum:
          R = euclidean(xyz[i], xyz[j])
          Rvdw = vdw_radii[z[i]] + vdw_radii[z[j]]
          if R < 0.9*Rvdw:
            conn_hbond[i, j] = conn_hbond[j, i] = 1

      return frags, conn, conn_frag, conn_frag_aux, conn_hbond

    def atoms_to_ric(self, atoms):

      angle_thresh = np.cos(175.0*np.pi/180.)

      coords = {}
      xyz = atoms.get_positions()
      xyzb = xyz * angs_to_bohr
      frags, conn, conn_frag, conn_frag_aux, conn_hbond = self.connectivity(atoms)
      natoms = len(atoms)

      total_conn = (conn + conn_frag + conn_hbond) > 0

      # bonds can be: covalent, interfragment, interfragment aux, or hbond
      for i, j in itertools.combinations(range(natoms), 2):
        if total_conn[i, j] or conn_frag_aux[i, j]:
          coords['stre_{}_{}'.format(i, j)] = Distance(i, j)

      # angles can be: covalent, interfragment, or hbond
      for i,j in itertools.permutations(range(natoms), 2):
        if total_conn[i, j]:
          for k in range(i+1, natoms):
            if total_conn[j, k]:
              check = self.checkangle(xyz[i], xyz[j], xyz[k])
              if not check: continue
              ang = Angle(i, j, k)
              if (np.cos(ang.value(xyzb))<angle_thresh):
                coords['linearbnd_{}_{}_{}_0'.format(i, j, k)] = LinearAngle(i, j, k, 0)
                coords['linearbnd_{}_{}_{}_1'.format(i, j, k)] = LinearAngle(i, j, k, 1)
              else:
                coords['bend_{}_{}_{}'.format(i, j, k)] = ang


      # torsions can be: covalent, interfragment, or hbond
      for i,j in itertools.permutations(range(natoms), 2):
        if total_conn[i, j]:
          for k in range(natoms):
            if total_conn[j, k] and i!=k and j!=k:
              for l in range(i+1, natoms): # l>i prevents double counting
                if total_conn[k, l] and i!=l and j!=l and k!=l and not total_conn[l, i]:
                  check = self.checktors(xyz[i], xyz[j], xyz[k], xyz[l])
                  if not check: continue
                  ang1 = Angle(i, j, k)
                  ang2 = Angle(j, k, l)
                  if np.abs(np.cos(ang1.value(xyzb))) > np.abs(angle_thresh): continue
                  if np.abs(np.cos(ang2.value(xyzb))) > np.abs(angle_thresh): continue
                  coords['tors_{}_{}_{}_{}'.format(i, j, k, l)] = Dihedral(i, j, k, l)

      # out-of-plane angle
      for b in range(natoms):
          b_neighbors = np.arange(natoms,)
          b_neighbors = b_neighbors[total_conn[b]>0]
          for a in b_neighbors:
            for c in b_neighbors:
              for d in b_neighbors:
                if a < c < d:
                  for i,j,k in sorted(list(itertools.permutations([a, c, d], 3))):
                    ang1 = Angle(b, i, j)
                    ang2 = Angle(i, j, k)
                    if np.abs(np.cos(ang1.value(xyzb))) > np.abs(angle_thresh): continue
                    if np.abs(np.cos(ang2.value(xyzb))) > np.abs(angle_thresh): continue
                    if np.abs(np.dot(ang1.normal_vector(xyzb), ang2.normal_vector(xyzb))) > angle_thresh:
                      coords["oop_{}_{}_{}_{}".format(b, i, j, k)] = OutOfPlane(b, i, j, k)
                      if natoms>4: break

      return coords

    def construct(self):

      coords1 = self.atoms_to_ric(self.atoms1)
      if self.atoms2 is None:
          return coords1

      coords2 = self.atoms_to_ric(self.atoms2)
      coords = {**coords1, **coords2}

      min_thresh = np.cos(120.*np.pi/180.)
      angle_thresh = np.cos(175.0*np.pi/180.)
      oop_thresh = np.abs(np.cos(85*np.pi/180.))
      lb_thresh = np.cos(45.*np.pi/180.)
      tors_thresh = np.abs(np.cos(175.0*np.pi/180.))

      # Check both ends for ill-defined torsions
      keys = list(coords.keys()).copy()
      to_delete = []
      to_add = {}
      xyzb1 = self.atoms1.get_positions() * angs_to_bohr
      xyzb2 = self.atoms2.get_positions() * angs_to_bohr
      for i, (name, coord) in enumerate(coords.items()):
          if "tors" in name:
              # check angle ABC and angle BCD for both geometries
              a, b, c, d = coord.a, coord.b, coord.c, coord.d
              ang1 = Angle(a, b, c)
              ang2 = Angle(b, c, d)
              if (np.abs(np.cos(ang1.value(xyzb1))) > tors_thresh) or \
                  (np.abs(np.cos(ang1.value(xyzb2))) > tors_thresh) or \
                  (np.abs(np.cos(ang2.value(xyzb1))) > tors_thresh) or \
                  (np.abs(np.cos(ang2.value(xyzb2))) > tors_thresh):
                      to_delete.append(name)
                      continue

      for k in set(to_delete):
          del(coords[k])

      for name, coord in to_add.items():
          coords[name] = coord

      # Remove angle coordinates displaced greater than pi or oop greater than pi/2
      self.coords = coords
      keys = list(coords.keys()).copy()
      to_delete = []
      to_add = {}
      q1 = self.q(self.atoms1.get_positions())
      q2 = self.q(self.atoms2.get_positions())
      dq = q2-q1
      for i, (name, coord) in enumerate(coords.items()):
          if ("bend" in name) and (np.cos(q1[i])<angle_thresh or np.cos(q2[i])<angle_thresh):
              if (np.abs(np.cos(q2[i]-q1[i]))<np.abs(min_thresh)):
                  to_delete.append(name)
          if ("oop" in name) and (np.cos(q1[i])<-oop_thresh or np.cos(q2[i])<-oop_thresh):
              to_delete.append(name)
          if ("tors" in name) and (np.cos(q1[i])<-tors_thresh or np.cos(q2[i])<-tors_thresh):
              to_delete.append(name)
              to_add['stre_{}_{}'.format(coord.a, coord.d)] = Distance(coord.a, coord.d)
          if ("linearbnd" in name) and ((np.cos(q1[i])<lb_thresh) or (np.cos(q2[i])<lb_thresh)):
              basecoord = name[:-2]
              to_delete.append(basecoord+"_0")
              to_delete.append(basecoord+"_1")
              to_add['bend_{}_{}_{}'.format(coord.a, coord.b, coord.c)] = Angle(coord.a, coord.b, coord.c)
          if ("linearbnd" in name):
              a, b, c = coord.a, coord.b, coord.c
              ang = Angle(a, b, c)
              angval1 = ang.value(xyzb1)
              angval2 = ang.value(xyzb2)
              if (np.abs(np.cos(angval2-angval1))>np.abs(min_thresh)):
                  basecoord = name[:-2]
                  to_delete.append(basecoord+"_0")
                  to_delete.append(basecoord+"_1")
                  to_add['bend_{}_{}_{}'.format(coord.a, coord.b, coord.c)] = Angle(coord.a, coord.b, coord.c)

      for k in set(to_delete):
          del(coords[k])

      for name, coord in to_add.items():
          coords[name] = coord

      # remove oop bends containing broken bonding centers
      keys = list(coords.keys()).copy()
      to_delete = []
      oop_keys = [i for i in keys if "oop" in i]
      tors_keys = [i for i in keys if "tors" in i]
      for oopk in oop_keys:
          coord = coords[oopk]
          if not ((oopk in coords1.keys()) and (oopk in coords2.keys())):
            to_delete.append(oopk)
            continue

      for k in set(to_delete):
          del(coords[k])

      keys = list(coords.keys()).copy()
      natoms = len(self.atoms1)
      tors_keys = [i for i in keys if "tors" in i]
      ntors = len(tors_keys)
      if ntors<1 and natoms>3:
        xyz1 = self.atoms1.get_positions()
        xyz2 = self.atoms2.get_positions()
        xyzb1 = xyz1 * angs_to_bohr
        xyzb2 = xyz2 * angs_to_bohr
        for i,j,k,l in list(itertools.permutations(range(natoms), 4)):
          check1 = self.checktors(xyz1[i], xyz1[j], xyz1[k], xyz1[l])
          check2 = self.checktors(xyz2[i], xyz2[j], xyz2[k], xyz2[l])
          check = check1*check2
          if not check: continue
          unique_perms = [p for p in itertools.permutations([i,j,k,l], 4) if p[-1]>p[0]]
          for a,b,c,d in unique_perms:
            ang1 = Angle(a, b, c)
            ang2 = Angle(b, c, d)
            tors = Dihedral(a, b, c, d)
            if (np.cos(tors.value(xyzb1))<-tors_thresh or np.cos(tors.value(xyzb2))<-tors_thresh):
              to_add['stre_{}_{}'.format(a, d)] = Distance(a, d)
              continue
            if np.abs(np.cos(ang1.value(xyzb1))) > tors_thresh: continue
            if np.abs(np.cos(ang1.value(xyzb2))) > tors_thresh: continue
            if np.abs(np.cos(ang2.value(xyzb1))) > tors_thresh: continue 
            if np.abs(np.cos(ang2.value(xyzb2))) > tors_thresh: continue 
            self.coords['tors_{}_{}_{}_{}'.format(a, b, c, d)] = Dihedral(a, b, c, d)
            ntors += 1
          if ntors > 0: break
      if ntors == 0 and natoms>3:
        coords = {}
        for i, j in itertools.combinations(range(natoms), 2):
          coords['stre_{}_{}'.format(i, j)] = Distance(i, j)
      return coords
