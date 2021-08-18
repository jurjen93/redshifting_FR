import numpy as np
from shapely.geometry import Polygon, Point, box
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

    def remove_background(self, image):
        """
        :return: masked background?
        """
        clipped_back = sigma_clip(image, 5)
        rms = np.std(clipped_back)
        cs = plt.contour(image, [1 * rms], colors='blue', linewidths=0.7)
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
        return image * mask

    def _get_polylist(self, image):
        """
        :param image: image data
        :param luminosity: luminosity or flux (True or False resp.)
        :return: list with polygons
        """
        im_source = self.remove_background(image)

        if self.FRI:
            self.poly_list = []
            if len(im_source)==0:
                return self
            cs = plt.contour(im_source, [self.fluxlim], colors='white', linewidths=0.7)
            cs_list = cs.collections[0].get_paths()
            for cs in cs_list:
                cs = cs.vertices
                if Polygon(cs).is_valid == True \
                    and Polygon(cs).contains(Point(self.opt_pos[0], self.opt_pos[1])) and Polygon(cs).area > 5:
                    self.poly_list.append(Polygon(cs))
        else:
            self.poly_lists = []
            if len(im_source) == 0:
                return self

            for i in np.exp(np.linspace(0, 60, 100))[::-1]:
                cs = plt.contour(image, [i/10 * self.fluxlim], linewidths=0.1, colors='black')
                cs_list = cs.collections[0].get_paths()
                poly_list = []
                while len(cs_list) >= 2:
                    for cs in cs_list:
                        if len(cs) < 3:
                            cs_list.remove(cs)
                    for cs in cs_list:
                        cs = cs.vertices
                        if Polygon(cs).is_valid == True:
                            poly_list.append(Polygon(cs))
                    # for poly in poly_list:
                    #     if Polygon(cs).area > 0.001: #poly.contains(Point(self.opt_pos[0], self.opt_pos[1])) == True
                    #         poly_list.remove(poly)

                    self.poly_lists+=[[poly_list[i], poly_list[j]] for i in range(len(poly_list)) for j in
                              range(i + 1, len(poly_list))]

                    return self

        return self


    def luminosity(self, dz=0.):
        """
        :param dz: delta redshift
        :return: Luminosity
        """
        z = self.redshift + dz
        if dz != 0:
            image = self.shift(dz)
        else:
            image = self.hdul.data

        self._get_polylist(image)

        if self.FRI:

            L = 0

            if len(self.poly_list) == 0:
                return "ERROR: no polygon"

            for poly in self.poly_list:
                perimeter = poly.exterior.coords
                x, y = np.meshgrid(np.arange(len(image)), np.arange(len(image)))
                x, y = x.flatten(), y.flatten()
                points = np.vstack((x, y)).T
                tup_verts = perimeter
                p = Path(tup_verts)
                grid = p.contains_points(points)
                mask = grid.reshape(len(image), len(image))
                im_part = image * mask
                surface_brightness = np.nansum(im_part)
                area = ((((1.5 * u.arcsec).to(u.rad).value * self.cosmo.luminosity_distance(z)) / (
                            1 + z) ** 2).to(u.pc).value) ** 2
                L += surface_brightness * area
            return self.Ftheta_to_Lpc(L, z)
        else:
            back = sigma_clip(image, 5)
            lim = 2*np.std(back)
            surface_brightness = np.nansum(image[image > lim])
            plt.imshow(np.where(image > lim, image, 0), norm=SymLogNorm(linthresh=self.fluxlim, vmin=0.1 * self.fluxlim, vmax=np.nanmax(self.hdul.data)),
                           cmap='magma')
            plt.show()
            area = ((((1.5 * u.arcsec).to(u.rad).value * self.cosmo.luminosity_distance(z)) / (1 + z) ** 2).to(u.pc).value) ** 2
            L = surface_brightness * area
            return self.Ftheta_to_Lpc(L, z)

    def size(self, dz, save_as=""):
        """
        :param save_as: give a name for the plot, if not given --> no plot
        :return: pixel size and physical size
        """
        z = self.redshift + dz
        image = self.shift(dz)

        self._get_polylist(image)

        bestcoords = [self.opt_pos, self.opt_pos]
        if self.FRI:
            if len(self.poly_list) == 0:
                return {'phys_size': "ERROR: no polygon", 'pix_size': "ERROR: no polygon"}
            cascadhull = cascaded_union(self.poly_list).convex_hull
            a = np.asarray(cascadhull.exterior.coords)
            mdist2 = 0
            for r in a:
                dist2 = np.sqrt((a[:, 0] - r[0]) ** 2 + (a[:, 1] - r[1]) ** 2) + \
                        np.sqrt((r[0] - self.opt_pos[0]) ** 2 + (r[1] - self.opt_pos[1]) ** 2) + \
                        np.sqrt((a[:, 0] - self.opt_pos[0]) ** 2 + (a[:, 1] - self.opt_pos[1]) ** 2)
                idist = np.argmax(dist2)
                mdist = dist2[idist]
                if mdist > mdist2:
                    mdist2 = mdist
                    bestcoords = (r, a[idist])
            pix_size = np.sqrt((bestcoords[0][0] - self.opt_pos[0]) ** 2 + (bestcoords[0][1] - self.opt_pos[1]) ** 2) + \
                       np.sqrt((bestcoords[1][0] - self.opt_pos[0]) ** 2 + (bestcoords[1][1] - self.opt_pos[1]) ** 2)
            phys_size = pix_size * (((1.5 * u.arcsec).to(u.rad).value * self.cosmo.luminosity_distance(z)) / (1 + z) ** 2).to(u.kpc).value


        else:
            if len(self.poly_lists) == 0:
                return {'phys_size': "ERROR: no polygon", 'pix_size': "ERROR: no polygon"}
            mdist2=0
            for poly_list in self.poly_lists:
                cascadhull = cascaded_union(poly_list).convex_hull
                # if cascadhull.contains(Point(self.opt_pos[0], self.opt_pos[1])) == True:
                point1, point2 = poly_list[0].representative_point(), poly_list[1].representative_point()
                mdist = np.sqrt((point1.x - self.opt_pos[0]) ** 2 + (point1.y - self.opt_pos[1]) ** 2) + \
                        np.sqrt((point2.x - self.opt_pos[0]) ** 2 + (point2.y - self.opt_pos[1]) ** 2) + \
                        np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)
                if mdist > mdist2:
                    mdist2 = mdist
                    bestcoords = [(point1.x, point1.y), (point2.x, point2.y)]


            pix_size = np.sqrt((bestcoords[0][0] - self.opt_pos[0]) ** 2 + (bestcoords[0][1] - self.opt_pos[1]) ** 2) + \
                       np.sqrt((bestcoords[1][0] - self.opt_pos[0]) ** 2 + (bestcoords[1][1] - self.opt_pos[1]) ** 2)

            phys_size = pix_size * (((1.5 * u.arcsec).to(u.rad).value * self.cosmo.luminosity_distance(z)) / (1 + z) ** 2).to(u.kpc).value


        if save_as:
            plt.close()
            if self.FRI:
                plt.imshow(image, norm=SymLogNorm(linthresh=self.fluxlim, vmin=0.1 * self.fluxlim, vmax=np.nanmax(self.hdul.data)),
                           cmap='magma')
            else:
                plt.imshow(image, norm=SymLogNorm(linthresh=10*self.fluxlim, vmin=0.1 * self.fluxlim, vmax=np.nanmax(self.hdul.data)),
                           cmap='magma')
            # x, y = cascadhull.exterior.xy
            # plt.plot(x, y)
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
            plt.axis('off')
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

    os.system('rm -rf ./test && rm ./data.csv && mkdir ./test')
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
            L00 = FR.luminosity(dz=0)
            L03 = FR.luminosity(dz=0.3)
            print(L00)
            print(L03)
    # header = ['source', 'type', 'z', 'dz', 'phys_size', 'pix_size', 'luminosity']
    # with open('data.csv', 'a+', encoding='UTF8') as f:
    #     writer = csv.writer(f)
    #     writer.writerow(header)
    #
    #     for ID in tqdm(catalogue.index):
    #         optical_position = SkyCoord(catalogue.loc[ID, 'ID_ra'], catalogue.loc[ID, 'ID_dec'], frame='icrs', unit=(u.degree, u.degree))
    #         middle_position = catalogue.loc[ID, 'RA'], catalogue.loc[ID, 'DEC']
    #         z = catalogue.loc[ID, 'z_best']
    #
    #         FRI = catalogue.loc[ID, 'FR1']
    #         FR = MeasureFR(fitsfile='cutout_fits/' + ID + '.fits', redshift=z, FRI=FRI, optical_position=optical_position, middle_position=middle_position)
    #
    #         if FRI:
    #             FR_type = 'FRI'
    #         else:
    #             FR_type = 'FRII'
    #
    #         try:
    #             s00 = FR.size(dz=0, save_as='test/'+ID+'_'+str(0.0)+'.png')
    #             L00 = FR.luminosity(dz=0)
    #             writer.writerow([ID, FR_type, z, 0, s00['phys_size'], s00['pix_size'], L00])
    #         except:
    #             pass
    #
    #         try:
    #             s01 = FR.size(dz=0.1, save_as='test/'+ID+'_'+str(0.1)+'.png')
    #             L01 = FR.luminosity(dz=0.1)
    #             writer.writerow([ID, FR_type, z, 0.1, s01['phys_size'], s01['pix_size'], L01])
    #         except:
    #             pass
    #
    #         try:
    #             s02 = FR.size(dz=0.2, save_as='test/'+ID+'_'+str(0.2)+'.png')
    #             L02 = FR.luminosity(dz=0.2)
    #             writer.writerow([ID, FR_type, z, 0.2, s02['phys_size'], s02['pix_size'], L02])
    #         except:
    #             pass
    #
    #         try:
    #             s03 = FR.size(dz=0.3, save_as='test/'+ID+'_'+str(0.3)+'.png')
    #             L03 = FR.luminosity(dz=0.3)
    #             writer.writerow([ID, FR_type, z, 0.3, s03['phys_size'], s03['pix_size'], L03])
    #         except:
    #             pass
    #
    #         try:
    #             s04 = FR.size(dz=0.4, save_as='test/'+ID+'_'+str(0.4)+'.png')
    #             L04 = FR.luminosity(dz=0.4)
    #             writer.writerow([ID, FR_type, z, 0.4, s04['phys_size'], s04['pix_size'], L04])
    #         except:
    #             pass