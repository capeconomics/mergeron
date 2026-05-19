Notes and Acknowlegements
=========================

FTC merger investigtions data published in 2004, 2007, 2008, and 2013 are provided in the module, along with constructed data for non-overlapping periods. For each period, as feasible, additional tables are constructed for, (a) enforcement and clearance counts for mergers in industries common to the contructed period and the base period; and (b) markets with no evidence on entry.

Source code for the function used to parse the source data in the relevant FTC publications for constructing the included dataset is included in the module, :code:`mergeron.core.ftc_merger_investigations_data`, but is inoperable by design. Users developing workarounds to reparse the data from the FTC publications will have to install additional software distributed under an incompatible license, and that license may impose additional restrictions on redistribution.

This package relies heavily on the Python packages, :code:`numpy` and :code:`matplotlib`, both for functionality and in the design of the APIs, with the exceptions that multi-valued input specifications are supplied as classes rather than as individual parameters. Classes for input-specification are defined using the :code:`attrs` package, making extensive use of the supplied validators and of custom validators.

Thanks to Prof. Ashwath Damodaran, New York University for his compilation of financial data, particularly gross margins, which are used to estimate an empirical margin distribution for potential merging firms.

On first attempt to specify data generation with empirical margin distribution, you may see the following message:

``WARNING: Could not establish secure connection to, https://pages.stern.nyu.edu/~adamodar/pc/datasets/margin.xls. Using bundled copy.``

If you wish to use use the latest data from Prof. Damodaran, please do the following:

#. download the file at the above URL to ~/mergeron/damodaran_margin_data.xls
#. delete the file ~/mergeron/damodaran_margin_data_serialized.zip
#. rerun your code
