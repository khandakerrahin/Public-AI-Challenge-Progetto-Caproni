def superimpose(image, list_of_damages, max_size=512):
    """
    param: image, image in which to superimpose damage/damages
    param: list_of_damages, list of paths of damages to superimpose
    param: max_size, size of the images (assuming height and width are the same)
    """
    m, n = max_size, max_size
    background_mask = Image.fromarray(np.zeros((n, m, 4), dtype='uint8')).convert('RGBA')

    while list_of_damages:
        curr_dmg = list_of_damages.pop()
        curr_dmg_kind = curr_dmg.split('/')[-2]
        if curr_dmg_kind == 'placeholder':
            curr_dmg = Image.fromarray(create_band(max_size)).convert('LA')
            coord_x, coord_y = 0, 0
        else:
            curr_dmg = Image.open(curr_dmg).convert('LA')
            coord_x, coord_y = choose_coord(max_size=512)
        # TODO Apply transformations to damage
        image.paste(curr_dmg, (coord_x, coord_y), mask=curr_dmg)
        # TODO curr damage has to be modified according to color map
        background_mask.paste(curr_dmg, (coord_x, coord_y), mask=curr_dmg)

    return image, background_mask
