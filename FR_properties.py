import numpy as np
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
from shapely.ops import cascaded_union
from astropy.wcs import WCS
from astropy.stats import sigma_clip
from matplotlib.path import Path
from astropy.table import Table
import astropy.units as u
from astropy.coordinates import SkyCoord
from matplotlib.colors import SymLogNorm
import warnings
from redshifting import RedShifting
from math import sin, pi
warnings.filterwarnings("ignore")

class MeasureFR(RedShifting):
    def __init__(self, fitsfile, redshift, FRI, optical_position, middle_position):
        self.optical_position = optical_position
        super().__init__(fitsfile, redshift, FRI, middle_position)

    @property
    def opt_pos(self):
        """
        :param header: header
        :param position: position in ra/dec
        :return: optical position
        """
        return WCS(self.current_header).wcs_world2pix(self.optical_position.ra, self.optical_position.dec, 1)

    def _get_polylist(self, image, luminosity=False):
        """
        :param image: image data
        :param luminosity: luminosity or flux (True or False resp.)
        :return: list with polygons
        """
        poly_list = []
        area1 = 0
        if self.FRI or luminosity:

            """
            We take this step if we are looking for FRIs or as first method for luminosities for FRIIs (2nd method below).
            
            Step 1: Clip data and remove background sources such that we keep one big source with the optical ID in it. 
                    We mask the rest.
            Step 2: Use this image to make contour plot of the leftover source. This is our polygon.
            """
            clipped_back = sigma_clip(image, 5)
            rms = np.std(clipped_back)
            cs = plt.contour(image, [1 * rms], colors='blue', linewidths=0.7)
            plt.close()
            cs_list = cs.collections[0].get_paths()
            correct_cs = []
            for cs in cs_list:
                if len(cs) > 2:
                    cs = cs.vertices
                    if Polygon(cs).contains(Point(self.opt_pos[0], self.opt_pos[1])):
                        correct_cs = cs
            if len(correct_cs) == 0:
                return []
            x, y = np.meshgrid(np.arange(len(image)), np.arange(len(image)))
            x, y = x.flatten(), y.flatten()
            points = np.vstack((x, y)).T
            tup_verts = correct_cs
            p = Path(tup_verts)
            grid = p.contains_points(points)
            mask = grid.reshape(len(image), len(image))
            im_source = image * mask

            cs = plt.contour(im_source, [self.fluxlim/2], colors='white', linewidths=0.7)
            cs_list = cs.collections[0].get_paths()
            poly_list = [Polygon(cs.vertices) for cs in cs_list if (Polygon(cs.vertices).is_valid == True
                                                                         and Polygon(cs.vertices).contains(Point(self.opt_pos[0], self.opt_pos[1]))
                                                                         and Polygon(cs.vertices).area>1)]
            if len(poly_list)>0:
                area1 = sum([Polygon(cs.vertices).area for cs in cs_list])
                if area1 < 15:
                    poly_list = []

        if self.FRI:
            return poly_list

        """       
        We take this step if we are looking for FRIIs or as second method for luminosities for FRIIs.

        Step 1: Find brightest points in image.
        Step 2: Take defining points of these brightest sources.
        Step 3: Obtain polygons from these source parts.
        """

        poly_lists = [] # list of polygon lists

        for i in np.exp(np.linspace(0, 60, 100))[::-1]:
            cs = plt.contour(image, [i/100 * self.fluxlim], linewidths=0.1, colors='black')
            plt.close()
            cs_list = cs.collections[0].get_paths()
            while len(cs_list) >= 2:
                for cs in cs_list:
                    if len(cs) < 3:
                        cs_list.remove(cs)
                sub_poly_list = [Polygon(cs.vertices) for cs in cs_list if Polygon(cs.vertices).is_valid]

                poly_lists += [[sub_poly_list[i], sub_poly_list[j]] for i in range(len(sub_poly_list)) for j in
                          range(i + 1, len(sub_poly_list))]
                break
            if len(poly_lists)>0: break


        if len(poly_lists)>0:
            for sub_poly_list in poly_lists:
                point1, point2 = sub_poly_list[0].representative_point(), sub_poly_list[1].representative_point()

            def polygon_contains(points, cs):
                for p in points:
                    if Polygon(cs.vertices).contains(p):
                        return True
                return False

            plt.close()

            cs = plt.contour(image, [np.std(self.hdul.data)], colors='white', linewidths=1)
            cs_list = cs.collections[0].get_paths()
            poly_list2 = [Polygon(cs.vertices) for cs in cs_list if (Polygon(cs.vertices).is_valid == True
                                         and (polygon_contains([point1, point2], cs)
                                              or Polygon(cs.vertices).contains(Point(self.opt_pos[0], self.opt_pos[1])))
                                         and Polygon(cs.vertices).area>3)]

            area2 = sum([Polygon(cs.vertices).area for cs in cs_list])

            if area2>area1:
                poly_list = poly_list2

        return poly_list


    def luminosity(self, dz=0., save_as=""):
        """
        :param dz: delta redshift
        :return: Luminosity
        """
        z = self.redshift + dz
        if dz != 0:
            image = self.shift(dz)
        else:
            image = self.hdul.data

        plt.close()
        poly_list = self._get_polylist(image, luminosity=True)

        L = 0

        if not poly_list or len(poly_list) == 0:
            return "ERROR: no polygon"

        mask = np.zeros(image.shape)
        for poly in poly_list:
            x, y = np.meshgrid(np.arange(len(image)), np.arange(len(image)))
            x, y = x.flatten(), y.flatten()
            points = np.vstack((x, y)).T
            grid = Path(poly.exterior.coords).contains_points(points)
            mask += grid.reshape(len(image), len(image))
        mask = np.where(mask>=1, np.ones(mask.shape), 0)
        im_source = image * mask
        surface_brightness = np.nansum(im_source*np.where(image > 0, 1, 0))
        area = ((((1.5 * u.arcsec).to(u.rad).value * self.cosmo.luminosity_distance(z)) / (
                    1 + z) ** 2).to(u.pc).value) ** 2
        L += surface_brightness * area
        L = self.Ftheta_to_Lpc(L, z)
        if save_as:
            plt.imshow(im_source,
                       norm=SymLogNorm(linthresh=self.fluxlim, vmin=0.1 * self.fluxlim, vmax=np.nanmax(self.hdul.data)),
                       cmap='magma')
            if self.FRI:
                plt.title('Physical size FRI (z='+str(round(z, 2))+'): ' + str(round(L,2))+' kpc')
            else:
                plt.title('Physical size FRII (z='+str(round(z, 2))+'): ' + str(round(L,2))+' kpc')
            plt.savefig(save_as)

        return L

    def size(self, dz, save_as=""):
        """
        :param save_as: give a name for the plot, if not given --> no plot
        :return: pixel size and physical size
        """
        z = self.redshift + dz
        image = self.shift(dz)

        poly_list = self._get_polylist(image)

        bestcoords = [self.opt_pos, self.opt_pos]

        if not poly_list or len(poly_list) == 0:
            return {'phys_size': "ERROR: no polygon", 'pix_size': "ERROR: no polygon"}

        cascadhull = cascaded_union(poly_list).convex_hull
        points = np.asarray(cascadhull.exterior.coords)
        mdist2=0

        dist_opt_points = np.sum(np.square(np.subtract(points, np.array(self.opt_pos))), axis=1)
        for n, r in enumerate(points):
            dist2 = np.sum(np.square(np.subtract(points, r)), axis=1)/2 + \
                    dist_opt_points[n] + \
                    dist_opt_points

            idist = np.argmax(dist2)
            mdist = dist2[idist]

            def angle_factor(angle):
                """
                calculate separation angle and return factor
                :param angle: angle of separation
                :return: angle factor from sin-rule
                """
                return sin(pi*angle/180)/sin(pi*(180-angle)/180/2)

            if self.FRI:
                angle_factor = angle_factor(30) # bigger than 60 degrees
            else:
                angle_factor = angle_factor(45) # bigger than 30 degrees

            if mdist > mdist2 \
                    and np.sum(np.square(np.subtract(points, r)), axis=1)[idist]>dist_opt_points[n]*angle_factor\
                    and np.sum(np.square(np.subtract(points, r)), axis=1)[idist]>dist_opt_points[idist]*angle_factor:
                mdist2 = mdist
                bestcoords = (r, points[idist])

        pix_size = np.sqrt((bestcoords[0][0] - self.opt_pos[0]) ** 2 + (bestcoords[0][1] - self.opt_pos[1]) ** 2) + \
                   np.sqrt((bestcoords[1][0] - self.opt_pos[0]) ** 2 + (bestcoords[1][1] - self.opt_pos[1]) ** 2)
        phys_size = pix_size * (((1.5 * u.arcsec).to(u.rad).value * self.cosmo.luminosity_distance(z)) / (1 + z) ** 2).to(u.kpc).value

        if save_as:
            plt.close()

            plt.imshow(image, norm=SymLogNorm(linthresh=self.fluxlim, vmin=self.fluxlim/10, vmax=np.nanmax(self.hdul.data)),
                           cmap='magma')

            plt.plot((bestcoords[0][0], self.opt_pos[0]), (bestcoords[0][1], self.opt_pos[1]),
                     color='green')
            plt.plot((bestcoords[1][0], self.opt_pos[0]), (bestcoords[1][1], self.opt_pos[1]),
                     color='green')
            plt.scatter(bestcoords[0][0], bestcoords[0][1], color='red', marker='*', zorder=2, s=100)
            plt.scatter(bestcoords[1][0], bestcoords[1][1], color='red', marker='*', zorder=2, s=100)
            plt.scatter(self.opt_pos[0], self.opt_pos[1], color='limegreen', marker='+', zorder=2,
                        s=100)
            if self.FRI:
                plt.title('Physical size FRI (z='+str(round(z, 2))+'): ' + str(round(phys_size, 2))+' kpc')
            else:
                plt.title('Physical size FRII (z='+str(round(z, 2))+'): ' + str(round(phys_size, 2))+' kpc')
            plt.savefig(save_as)
            plt.tight_layout()
            plt.close()

        return {'pix_size': pix_size, 'phys_size': phys_size}


if __name__ == '__main__':
    import csv
    from astropy.utils.data import get_pkg_data_filename
    from tqdm import tqdm
    import os

    hdul = get_pkg_data_filename('Mingo19_LoMorph_Cat.fits')
    hdul2 = get_pkg_data_filename('LOFAR_HBA_T1_DR1_merge_ID_optical_f_v1.2b_restframe.fits')

    catalogue = Table.read(hdul, hdu=1).to_pandas().set_index('Source_Name')
    catalogue.index = catalogue.index.str.decode('utf-8')  # decode

    # Selecting only the FR1 and FR2 from the catalogue
    # Deleting first two RGs (weird size)
    catalogue = catalogue[(catalogue['FR1'] | catalogue['FR2'])
                          & (catalogue.index != 'ILTJ142930.70+544406.2')
                          & (catalogue.index != 'ILTJ134313.28+560016.9')]

    large_catalogue = Table.read(hdul2, hdu=1).to_pandas().set_index('Source_Name')
    large_catalogue.index = large_catalogue.index.str.decode('utf-8')  # decode

    catalogue = catalogue.merge(large_catalogue[['z_best', 'Mosaic_ID', 'LGZ_Size', 'ID_ra', 'ID_dec', 'Total_flux', 'LGZ_Size']], how="inner",
                                left_index=True, right_index=True)

    del large_catalogue

    # Removing all the sources form the P41Hetdex because this mosaic is not present on the strw database.
    catalogue = catalogue[catalogue['Mosaic_ID'] != b'P41Hetdex']
    test_data = ['ILTJ114756.11+535822.2', 'ILTJ114747.49+480720.4']

    # os.system('rm -rf ./test && rm ./data.csv && mkdir test')
    # for ID in tqdm(catalogue.index):
    #     optical_position = SkyCoord(catalogue.loc[ID, 'ID_ra'], catalogue.loc[ID, 'ID_dec'], frame='icrs', unit=(u.degree, u.degree))
    #     middle_position = catalogue.loc[ID, 'RA'], catalogue.loc[ID, 'DEC']
    #     z = catalogue.loc[ID, 'z_best']
    #
    #     FRI = catalogue.loc[ID, 'FR1']
    #     FR = MeasureFR(fitsfile='cutout_fits/' + ID + '.fits', redshift=z, FRI=FRI, optical_position=optical_position, middle_position=middle_position)
    #
    #     if FRI:
    #         FR_type = 'FRI'
    #     else:
    #         FR_type = 'FRII'
    #     print(FR.luminosity(dz=0, save_as='test/' + ID + '_lum_' + str(0.0) + '.png'))
    #     print(FR.luminosity(dz=0.2, save_as='test/' + ID + '_lum_' + str(0.2) + '.png'))
    #     print(FR.luminosity(dz=0.3, save_as='test/' + ID + '_lum_' + str(0.3) + '.png'))
        # FR.size(dz=0, save_as='test/' + ID + '_' + str(0.0) + '.png')
        # FR.size(dz=0.2, save_as='test/' + ID + '_' + str(0.2) + '.png')
        # FR.size(dz=0.3, save_as='test/' + ID + '_' + str(0.3) + '.png')

    header = ['source', 'type', 'z', 'dz', 'phys_size', 'pix_size', 'luminosity']


    with open('data.csv', 'a+', encoding='UTF8') as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for ID in tqdm(catalogue.index):
            optical_position = SkyCoord(catalogue.loc[ID, 'ID_ra'], catalogue.loc[ID, 'ID_dec'], frame='icrs', unit=(u.degree, u.degree))
            middle_position = catalogue.loc[ID, 'RA'], catalogue.loc[ID, 'DEC']
            z = catalogue.loc[ID, 'z_best']

            FRI = catalogue.loc[ID, 'FR1']
            FR = MeasureFR(fitsfile='cutout_fits/' + ID + '.fits', redshift=z, FRI=FRI, optical_position=optical_position, middle_position=middle_position)

            if FRI:
                FR_type = 'FRI'
            else:
                FR_type = 'FRII'

            try:
                s00 = FR.size(dz=0, save_as='test/'+ID+'_'+str(0.0)+'.png')
                L00 = FR.luminosity(dz=0, save_as='test/'+ID+'_lum_'+str(0.0)+'.png')
                writer.writerow([ID, FR_type, z, 0, s00['phys_size'], s00['pix_size'], L00])
            except:
                pass

            try:
                s01 = FR.size(dz=0.1, save_as='test/'+ID+'_'+str(0.1)+'.png')
                L01 = FR.luminosity(dz=0.1, save_as='test/'+ID+'_lum_'+str(0.0)+'.png')
                writer.writerow([ID, FR_type, z, 0.1, s01['phys_size'], s01['pix_size'], L01])
            except:
                pass

            try:
                s02 = FR.size(dz=0.2, save_as='test/'+ID+'_'+str(0.2)+'.png')
                L02 = FR.luminosity(dz=0.2, save_as='test/'+ID+'_lum_'+str(0.2)+'.png')
                writer.writerow([ID, FR_type, z, 0.2, s02['phys_size'], s02['pix_size'], L02])
            except:
                pass

            try:
                s03 = FR.size(dz=0.3, save_as='test/'+ID+'_'+str(0.3)+'.png')
                L03 = FR.luminosity(dz=0.3, save_as='test/'+ID+'_lum_'+str(0.3)+'.png')
                writer.writerow([ID, FR_type, z, 0.3, s03['phys_size'], s03['pix_size'], L03])
            except:
                pass

            try:
                s04 = FR.size(dz=0.4, save_as='test/'+ID+'_'+str(0.4)+'.png')
                L04 = FR.luminosity(dz=0.4, save_as='test/'+ID+'_lum_'+str(0.4)+'.png')
                writer.writerow([ID, FR_type, z, 0.4, s04['phys_size'], s04['pix_size'], L04])
            except:
                pass