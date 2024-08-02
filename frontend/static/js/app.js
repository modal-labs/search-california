document.addEventListener("alpine:init", () => {
  Alpine.data("searchForm", () => ({
    get lat() {
      return Alpine.store("searchForm").lat;
    },
    set lat(value) {
      Alpine.store("searchForm").lat = value;
    },
    get lon() {
      return Alpine.store("searchForm").lon;
    },
    set lon(value) {
      Alpine.store("searchForm").lon = value;
    },
    get since() {
      return Alpine.store("searchForm").since;
    },
    set since(value) {
      Alpine.store("searchForm").since = value;
    },
    get before() {
      return Alpine.store("searchForm").before;
    },
    set before(value) {
      Alpine.store("searchForm").before = value;
    },
    loading: false,
    errorMessage: "",

    init() {
      this.postGeoSearch();
    },

    async postGeoSearch() {
      if (
        isNaN(this.lat) ||
        isNaN(this.lon) ||
        this.lat < -90 ||
        this.lat > 90 ||
        this.lon > 180 ||
        this.lon < -180
      ) {
        this.errorMessage = "Please enter valid latitude and longitude values.";
        return;
      }

      this.loading = true;
      this.errorMessage = "";

      const queryParams = new URLSearchParams();
      if (this.since) queryParams.append("since", this.since);
      if (this.before) queryParams.append("before", this.before);

      try {
        const response = await fetch(
          `https://modal-labs--clay-mongo-client-geo-search.modal.run?${queryParams.toString()}`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              lat: parseFloat(this.lat),
              lon: parseFloat(this.lon),
            }),
          }
        );

        if (response.ok) {
          const data = await response.json();
          Alpine.store("imageGrid").searchResults = data.results;
        } else {
          this.errorMessage = `Error: ${response.statusText}`;
        }
      } catch (error) {
        this.errorMessage = `Error: ${error.message}`;
      }
      this.loading = false;
    },
  }));

  Alpine.data("imageGrid", () => ({
    get searchResults() {
      return Alpine.store("imageGrid").searchResults;
    },
    loading: false,
    errorMessage: "",

    async onImageClick(item) {
      this.loading = true;
      this.errorMessage = "";

      if (item.lat)
        Alpine.store("searchForm").lat = parseFloat(item.lat.toFixed(1));
      if (item.lon)
        Alpine.store("searchForm").lon = parseFloat(item.lon.toFixed(1));

      const queryParams = new URLSearchParams();
      if (Alpine.store("searchForm").since)
        queryParams.append("since", Alpine.store("searchForm").since);
      if (Alpine.store("searchForm").before)
        queryParams.append("before", Alpine.store("searchForm").before);

      try {
        const response = await fetch(
          `https://modal-labs--clay-mongo-client-vector-search.modal.run?${queryParams.toString()}`,
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ vector: item.vector }),
          }
        );

        if (response.ok) {
          const data = await response.json();
          Alpine.store("imageGrid").searchResults = data.results;
        } else {
          const errorData = await response.json();
          this.errorMessage = `Error: ${errorData.message}`;
        }
      } catch (error) {
        this.errorMessage = `Error: ${error.message}`;
      }
      this.loading = false;
    },

    onImageError(event) {
      event.target.closest("li").style.display = "none";
      event.target.closest("li").style.margin = 0;
    },
  }));

  Alpine.store("searchForm", {
    since: "",
    before: "",
    lat: 37.8,
    lon: -122.4,
  });

  Alpine.store("imageGrid", {
    searchResults: [],
  });
});
