"""Bounded scene estimates, separate from facts about detected objects."""

# Display wording never comes from unrestricted model prose.
SETTINGS = {
    "unclear": ("an unclear setting", "unclear"),
    "home": ("a home or living area", "indoor"),
    "kitchen": ("a kitchen", "indoor"),
    "bedroom": ("a bedroom", "indoor"),
    "bathroom": ("a bathroom", "indoor"),
    "office": ("an office or workspace", "indoor"),
    "classroom": ("a classroom or study room", "indoor"),
    "restaurant": ("a restaurant or dining area", "indoor"),
    "ballroom": ("a ballroom or event hall", "indoor"),
    "shop": ("a shop or retail area", "indoor"),
    "street": ("a street or roadside area", "outdoor"),
    "park": ("a park or garden", "outdoor"),
    "sports_area": ("a sports or recreation area", "unclear"),
    "station": ("a transport station or platform", "unclear"),
    "farm": ("a farm or pasture", "outdoor"),
    "wildlife_area": ("a wildlife area or animal enclosure", "outdoor"),
    "waterfront": ("a beach or waterfront", "outdoor"),
    "snow_area": ("a snowy recreation area", "outdoor"),
    "airport": ("an airport or airfield", "unclear"),
    "vehicle_interior": ("the inside of a vehicle", "indoor"),
}


def environment_text(environment, setting, source="LLM"):
    if setting != "unclear":
        return f"Looks like {SETTINGS[setting][0]} ({source} estimate; unconfirmed)."
    return {
        "indoor": f"Appears indoors; room type is unclear ({source} estimate).",
        "outdoor": f"Appears outdoors; place type is unclear ({source} estimate).",
        "unclear": "Unclear; the available evidence does not establish the setting.",
    }[environment]
